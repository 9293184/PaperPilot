"""回归测试：覆盖已修复的缺陷与文献互通服务。"""

from __future__ import annotations

import json

import pytest

from app.db import session
from app.db.sqlite import purge_database_data
from app.models import FolderCreate, PaperCreate, PaperUpdate
from app.services import interop_service
from app.services.folder_service import batch_import_files, batch_import_papers, create_folder
from app.services.paper_service import create_paper, get_paper, update_paper, upsert_attachment_file


class TestPurgeDatabase:
    def test_purge_does_not_raise_and_clears(self, temp_workspace):
        create_paper(PaperCreate(title="p1", status="uploaded"))
        create_paper(PaperCreate(title="p2", status="uploaded"))
        create_folder(FolderCreate(name="f1"))

        # 不应因 sqlite_sequence 不存在而抛异常
        purge_database_data()

        with session() as conn:
            for table in ("papers", "folders", "paper_analysis", "attachments"):
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert count == 0, table


class TestUpdatePaperMissing:
    def test_update_missing_paper_raises_value_error(self, temp_workspace):
        with pytest.raises(ValueError):
            update_paper("does-not-exist", PaperUpdate(title="x"))


class TestUpdatePaperMetadataOnFreshPaper:
    def test_metadata_insert_works_when_no_metadata_row(self, temp_workspace):
        """回归：metadata 插入语句参数个数不匹配，导致新论文写元数据必失败。"""
        paper_id = create_paper(PaperCreate(title="T", status="uploaded"))
        update_paper(paper_id, PaperUpdate(doi="10.1/x", keywords="a,b", year="2024", source="J"))

        detail = get_paper(paper_id)
        assert detail is not None
        assert detail.metadata is not None
        assert detail.metadata.doi == "10.1/x"
        assert detail.metadata.keywords == "a,b"


class TestBatchImport:
    def test_batch_import_creates_papers(self, temp_workspace, monkeypatch):
        """回归：create_paper 返回 str，原代码按 dict 取值导致导入必失败。"""
        # 屏蔽后台分析线程，避免测试期间产生副作用
        monkeypatch.setattr("app.services.auto_parse_and_analyze", lambda *a, **k: None)

        folder = create_folder(FolderCreate(name="导入目录"))
        results = batch_import_papers(folder.id, [("paper_a.pdf", b"%PDF-1.4 fake")])

        assert len(results) == 1
        assert results[0]["success"] is True, results[0]["error"]
        assert results[0]["paper_id"]

        detail = get_paper(results[0]["paper_id"])
        assert detail is not None
        assert detail.title == "paper_a"

    def test_batch_import_files_from_paths(self, temp_workspace, tmp_path, monkeypatch):
        """新入口直接接收已落盘文件，不再把全部文件读进内存。"""
        monkeypatch.setattr("app.services.auto_parse_and_analyze", lambda *a, **k: None)

        folder = create_folder(FolderCreate(name="导入目录2"))
        src = tmp_path / "paper_b.pdf"
        src.write_bytes(b"%PDF-1.4 fake")

        results = batch_import_files(folder.id, [("paper_b.pdf", src)])

        assert len(results) == 1
        assert results[0]["success"] is True, results[0]["error"]
        detail = get_paper(results[0]["paper_id"])
        assert detail is not None and detail.title == "paper_b"


class TestAttachmentStatus:
    def test_uploading_non_original_does_not_reset_status(self, temp_workspace, tmp_path):
        """回归：上传翻译件/对应件不应把已完成论文的状态回退为 parsed。"""
        paper_id = create_paper(PaperCreate(title="P", status="uploaded"))
        with session() as conn:
            conn.execute("UPDATE papers SET status = 'done' WHERE id = ?", (paper_id,))

        src = tmp_path / "translated.pdf"
        src.write_bytes(b"%PDF-1.4 fake")
        upsert_attachment_file(paper_id, "translated", str(src), "translated.pdf")

        detail = get_paper(paper_id)
        assert detail is not None
        assert detail.status == "done"

    def test_uploading_original_advances_status(self, temp_workspace, tmp_path):
        paper_id = create_paper(PaperCreate(title="P", status="uploaded"))
        src = tmp_path / "original.pdf"
        src.write_bytes(b"%PDF-1.4 fake")
        upsert_attachment_file(paper_id, "original", str(src), "original.pdf")

        detail = get_paper(paper_id)
        assert detail is not None
        assert detail.status == "parsed"


class TestRestoreValidation:
    def test_restore_rejects_non_zip(self, temp_workspace, tmp_path):
        from app.services.backup_service import restore_full_backup

        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"definitely not a zip")
        with pytest.raises(ValueError):
            restore_full_backup(bad)

    def test_restore_rejects_zip_without_manifest(self, temp_workspace, tmp_path):
        import zipfile

        from app.services.backup_service import restore_full_backup

        path = tmp_path / "nomanifest.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("some.txt", "hello")
        with pytest.raises(ValueError):
            restore_full_backup(path)


class TestPaperLock:
    def test_analysis_lock_is_exclusive(self):
        from app.core.paper_lock import release, try_acquire

        assert try_acquire("p1") is True
        assert try_acquire("p1") is False  # 已有任务在跑
        release("p1")
        assert try_acquire("p1") is True
        release("p1")


class TestChatContextScope:
    def test_full_text_prefers_mineru_over_metadata(self, temp_workspace):
        """回归：按 updated_at 取会让「改过元数据」把 metadata 行顶到最前，
        导致问答上下文静默降级为摘要。"""
        from app.services import chat_service

        pid = create_paper(PaperCreate(title="P", status="uploaded"))
        with session() as conn:
            conn.execute(
                "INSERT INTO paper_texts (id, paper_id, text_scope, raw_text, parse_status) VALUES ('m1', ?, 'metadata', 'SHORT_ABSTRACT', 'done')",
                (pid,),
            )
            conn.execute(
                "INSERT INTO paper_texts (id, paper_id, text_scope, raw_text, parse_status) VALUES ('m2', ?, 'mineru', 'FULL_TEXT_BODY', 'done')",
                (pid,),
            )
            # 让 metadata 行看起来「更新更晚」
            conn.execute("UPDATE paper_texts SET updated_at = '2099-01-01 00:00:00' WHERE id = 'm1'")

        assert chat_service._get_paper_full_text(pid) == "FULL_TEXT_BODY"


class TestAnnotationsUpsert:
    def test_saving_twice_updates_single_row(self, temp_workspace):
        from app.services.paper_service import get_paper_annotations, save_paper_annotations

        pid = create_paper(PaperCreate(title="P", status="uploaded"))
        save_paper_annotations(pid, "original", [{"id": 1}])
        save_paper_annotations(pid, "original", [{"id": 2}])

        with session() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM paper_annotations WHERE paper_id = ?", (pid,)
            ).fetchone()[0]
        assert count == 1
        assert get_paper_annotations(pid, "original")["annotations"] == [{"id": 2}]


class TestSqliteWal:
    def test_checkpoint_does_not_raise(self, temp_workspace):
        from app.db.sqlite import checkpoint_database

        checkpoint_database()

    def test_wal_mode_enabled(self, temp_workspace):
        with session() as conn:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"


class TestFuzzySearchBounds:
    def test_exact_match_covers_full_text(self, temp_workspace):
        from app.services.paper_service import _is_fuzzy_match

        long_text = ("x" * 50_000) + " transformer"
        assert _is_fuzzy_match(long_text, "transformer") is True

    def test_typo_still_matches_near_start(self, temp_workspace):
        from app.services.paper_service import _is_fuzzy_match

        assert _is_fuzzy_match("attention is all you need", "atention") is True


class TestInitialSeed:
    @pytest.fixture(autouse=True)
    def _no_background_analysis(self, monkeypatch):
        # 播种会像上传一样触发后台分析；测试里屏蔽掉，避免真实解析与网络调用
        monkeypatch.setattr("app.services.seed_service.run_analysis_exclusive", lambda *a, **k: None)

    def test_seeds_papers_with_pdfs_and_only_once(self, temp_workspace):
        """首次启动写入 3 篇内置文献并放置 PDF；重复调用不重复插入。"""
        from app.services.seed_service import seed_initial_library

        assert seed_initial_library() == 3
        assert seed_initial_library() == 0  # 标记文件已存在

        with session() as conn:
            assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3
            originals = conn.execute(
                "SELECT COUNT(*) FROM attachments WHERE attachment_type = 'original'"
            ).fetchone()[0]
            assert originals == 3

        # PDF 已按「上传」的规范布局落盘
        for pid in ("paper-demo-0001", "paper-demo-0002", "paper-demo-0003"):
            pdf = temp_workspace / "storage" / pid / "original.pdf"
            assert pdf.exists(), pid
            assert pdf.stat().st_size > 0, pid

    def test_seeded_papers_look_like_uploads(self, temp_workspace):
        """内置文献应跟「刚上传 PDF」一致：有原件、元数据完整。"""
        from app.services.seed_service import seed_initial_library

        seed_initial_library()
        detail = get_paper("paper-demo-0001")
        assert detail is not None
        assert [a.attachment_type for a in detail.attachments] == ["original"]
        assert detail.metadata is not None
        assert detail.metadata.doi

    def test_does_not_seed_when_library_not_empty(self, temp_workspace):
        """库内已有论文时不插入内置数据（例如从备份恢复后）。"""
        from app.services.seed_service import seed_initial_library

        create_paper(PaperCreate(title="已存在的论文", status="uploaded"))
        assert seed_initial_library() == 0
        with session() as conn:
            assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1


class TestAnalysisFallback:
    def test_fallback_is_empty_not_placeholder_text(self):
        """回归：分析失败时不得填入「Demo 降级输出」之类的占位文案。

        之前会把八个维度填满占位文字，状态虽是 failed，用户却会误以为分析成功。
        """
        from app.core.analysis import DEFAULT_FIELDS, _fallback_analysis

        result = _fallback_analysis()
        assert set(result) == set(DEFAULT_FIELDS)
        assert all(value == "" for value in result.values()), result

        joined = " ".join(result.values())
        for word in ("Demo", "demo", "降级", "当前版本"):
            assert word not in joined, word


class TestWorkspaceDirOverride:
    def test_env_var_moves_workspace_and_db(self, monkeypatch, tmp_path):
        """部署时用 PAPERPILOT_WORKSPACE_DIR 把运行时数据指到挂载卷。"""
        from app.core.config import Settings

        target = tmp_path / "mounted-volume"
        monkeypatch.setenv("PAPERPILOT_WORKSPACE_DIR", str(target))

        s = Settings()
        assert s.workspace_dir == target.resolve()
        # db_path 的默认值是在类定义时算的，必须一并重算
        assert s.db_path == target.resolve() / "paperreading.db"
        # 内置数据目录不受影响（schema/seed/seed_pdfs 仍来自仓库）
        assert (s.data_dir / "schema.sql").exists()
        assert (s.data_dir / "seed_pdfs").is_dir()

    def test_no_env_var_keeps_default(self, monkeypatch):
        from app.core.config import Settings

        monkeypatch.delenv("PAPERPILOT_WORKSPACE_DIR", raising=False)
        s = Settings()
        assert s.workspace_dir.name == "workspace"
        assert s.db_path == s.workspace_dir / "paperreading.db"

    def test_missing_pdf_degrades_to_terminal_status(self, temp_workspace, monkeypatch):
        """内置 PDF 缺失时降级为终态，不能卡在「正在分析」。"""
        from app.services import seed_service
        from app.services.paper_service import PAPER_STATUS_IMPORTED
        from app.services.seed_service import seed_initial_library

        monkeypatch.setattr(seed_service, "_SEED_PDF_DIR", temp_workspace / "nonexistent")
        seed_initial_library()

        detail = get_paper("paper-demo-0001")
        assert detail is not None
        assert detail.status == PAPER_STATUS_IMPORTED
        assert detail.attachments == []


class TestInteropService:
    def test_import_bibtex_then_export(self, temp_workspace):
        bib = """
@article{k1,
  title = {Attention is All You Need},
  author = {Vaswani, Ashish and Shazeer, Noam},
  journal = {NeurIPS},
  year = {2017},
  doi = {10.5555/1},
  keywords = {transformer, attention}
}
"""
        result = interop_service.import_text(bib, filename="a.bib")
        assert result["format"] == "bibtex"
        assert result["imported"] == 1
        assert result["failed"] == 0

        records = interop_service.export_records()
        assert len(records) == 1
        rec = records[0]
        assert rec["title"] == "Attention is All You Need"
        assert rec["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
        assert rec["doi"] == "10.5555/1"
        assert set(rec["keywords"]) == {"transformer", "attention"}

    def test_import_is_idempotent_with_dedupe(self, temp_workspace):
        bib = "@article{k1, title={Same Paper}, author={A, B}, year={2020}}"
        first = interop_service.import_text(bib, filename="a.bib")
        second = interop_service.import_text(bib, filename="a.bib")
        assert first["imported"] == 1
        assert second["imported"] == 0 and second["skipped"] == 1

    def test_import_ris_and_export_all_formats(self, temp_workspace):
        ris = "TY  - JOUR\nTI  - Hello World\nAU  - Doe, John\nPY  - 2021\nER  - \n"
        result = interop_service.import_text(ris, filename="a.ris")
        assert result["imported"] == 1

        for fmt in ("csljson", "bibtex", "ris"):
            text = interop_service.export_text(fmt)
            assert text.strip(), fmt
        items = json.loads(interop_service.export_text("csljson"))
        assert items[0]["title"] == "Hello World"

    def test_import_assigns_tags_and_folder(self, temp_workspace):
        bib = "@article{k1, title={Tagged}, author={A, B}, year={2020}, keywords={ml, nlp}}"
        result = interop_service.import_text(bib, filename="a.bib")
        assert result["imported"] == 1

        with session() as conn:
            tag_names = {
                row["name"]
                for row in conn.execute(
                    "SELECT t.name FROM tags t JOIN paper_tags pt ON pt.tag_id = t.id"
                ).fetchall()
            }
        assert tag_names == {"ml", "nlp"}

    def test_import_chinese_title_does_not_fill_title_en(self, temp_workspace):
        """导入不应把中文标题写进 title_en（语言回填交给 get_paper）。"""
        bib = "@article{k1, title={预训练模型综述}, author={{张三}}, year={2020}}"
        result = interop_service.import_text(bib, filename="a.bib")
        assert result["imported"] == 1

        detail = get_paper(result["items"][0]["paper_id"])
        assert detail is not None
        assert detail.title == "预训练模型综述"
        assert detail.title_en == ""

    def test_imported_paper_status_is_terminal_not_analyzing(self, temp_workspace):
        """回归：导入的论文只有题录、没有 PDF，状态不能是 'uploaded'。

        否则前端会一直显示「等待解析 / 正在分析」，进度永远停在 0%。
        """
        from app.services.paper_service import PAPER_STATUS_IMPORTED

        bib = "@article{k1, title={Only Metadata}, author={A, B}, year={2020}}"
        result = interop_service.import_text(bib, filename="a.bib")
        detail = get_paper(result["items"][0]["paper_id"])
        assert detail is not None
        assert detail.status == PAPER_STATUS_IMPORTED
        # 没有任何附件 → 不存在可运行的分析
        assert detail.attachments == []
