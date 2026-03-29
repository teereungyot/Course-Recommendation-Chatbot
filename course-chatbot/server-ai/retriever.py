import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import CHROMA_ROOT, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


DEPARTMENT_PROGRAM_MAP = {
    "CS": "Computer Science",
    "MWCS": "Mathematics with Computer Science",
    "SDA": "Statistical Data Science and Analytics",
    "AS": "Applied Statistics",
    "ASB": "Business Statistics and Actuarial Science",
    "BT": "Biotechnology",
    "FST": "Food Science and Technology",
    "EST": "Environmental Science and Technology",
    "HBS": "Health and Beauty Science",
}

PROGRAM_ALIASES = {
    "Computer Science": {
        "computer science",
        "cs",
        "computer science program",
        "department of computer science",
        "b.sc. computer science",
    },
    "Mathematics with Computer Science": {
        "mathematics with computer science",
        "mwcs",
        "math with computer science",
        "mathematics and computer science",
    },
    "Statistical Data Science and Analytics": {
        "statistical data science and analytics",
        "sda",
        "data science and analytics",
    },
    "Applied Statistics": {
        "applied statistics",
        "as",
    },
    "Business Statistics and Actuarial Science": {
        "business statistics and actuarial science",
        "asb",
        "actuarial science",
        "business statistics",
    },
    "Biotechnology": {
        "biotechnology",
        "bt",
    },
    "Food Science and Technology": {
        "food science and technology",
        "fst",
        "food science",
    },
    "Environmental Science and Technology": {
        "environmental science and technology",
        "est",
        "environmental science",
    },
    "Health and Beauty Science": {
        "health and beauty science",
        "hbs",
        "beauty science",
        "cosmetic science",
    },
}

DEPARTMENT_HINT_KEYWORDS = {
    "CS": {
        "programming", "coding", "software", "web", "frontend", "backend", "full stack",
        "database", "ai", "machine learning", "data structure", "algorithm", "computer",
        "ระบบ", "เขียนโปรแกรม", "พัฒนาเว็บ", "ซอฟต์แวร์", "ฐานข้อมูล", "โครงสร้างข้อมูล",
        "อัลกอริทึม", "ปัญญาประดิษฐ์", "คอมพิวเตอร์", "วิเคราะห์ข้อมูล"
    },
    "MWCS": {
        "mathematics", "math", "proof", "modeling", "algorithm", "computer",
        "คณิตศาสตร์", "พิสูจน์", "แบบจำลอง", "อัลกอริทึม", "คอมพิวเตอร์"
    },
    "SDA": {
        "data", "analytics", "analysis", "machine learning", "statistics", "visualization",
        "ข้อมูล", "วิเคราะห์ข้อมูล", "สถิติ", "การพยากรณ์", "แดชบอร์ด"
    },
    "AS": {
        "statistics", "sampling", "probability", "analysis", "survey",
        "สถิติ", "ความน่าจะเป็น", "การสุ่มตัวอย่าง", "การวิเคราะห์"
    },
    "ASB": {
        "insurance", "actuary", "risk", "finance", "business analytics",
        "ประกันภัย", "คณิตศาสตร์ประกันภัย", "ความเสี่ยง", "การเงิน", "ธุรกิจ"
    },
    "BT": {
        "biology", "biotech", "lab", "experiment", "microbiology",
        "ชีววิทยา", "ชีวเทคโนโลยี", "ห้องทดลอง", "ทดลอง", "จุลชีววิทยา"
    },
    "FST": {
        "food", "quality control", "production", "safety", "lab",
        "อาหาร", "ความปลอดภัยอาหาร", "ควบคุมคุณภาพ", "การผลิต", "ห้องทดลอง"
    },
    "EST": {
        "environment", "pollution", "waste", "sustainability",
        "สิ่งแวดล้อม", "มลพิษ", "ของเสีย", "ความยั่งยืน"
    },
    "HBS": {
        "cosmetic", "beauty", "skincare", "product safety", "lab",
        "เครื่องสำอาง", "ความงาม", "สกินแคร์", "ความปลอดภัยผลิตภัณฑ์", "ห้องทดลอง"
    },
}

CROSS_DOMAIN_MISMATCH_KEYWORDS = {
    "CS": {
        "chemistry", "chemical", "cosmetic", "food science", "biotech", "biology", "lab experiment",
        "เคมี", "สารเคมี", "ห้องทดลอง", "ทดลองในแล็บ", "ชีววิทยา", "ชีวเคมี", "เครื่องสำอาง", "อาหาร"
    },
    "BT": {
        "web development", "frontend", "backend", "full stack", "software engineer",
        "พัฒนาเว็บ", "เขียนโปรแกรมเว็บ", "ฟรอนต์เอนด์", "แบ็กเอนด์", "วิศวกรซอฟต์แวร์"
    },
    "FST": {
        "frontend", "backend", "web application", "mobile app",
        "ฟรอนต์เอนด์", "แบ็กเอนด์", "เว็บแอป", "แอปมือถือ"
    },
    "HBS": {
        "frontend", "backend", "database", "software development",
        "ฟรอนต์เอนด์", "แบ็กเอนด์", "ฐานข้อมูล", "พัฒนาซอฟต์แวร์"
    },
}

MIN_RELEVANCE_SCORE = 0.12
MIN_FINAL_SCORE = 1.5


def _to_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                pass

        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]

        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]

        return [text]

    return [str(value).strip()]


# def get_embedding():
#     return OllamaEmbeddings(model="nomic-embed-text-v2-moe")
def get_embedding():
    return OllamaEmbeddings(
        model=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9ก-๙]+", " ", _normalize_space(text)).strip()


def _contains_any(text: str, keywords: Set[str]) -> bool:
    normalized = _compact_text(text)
    return any(_compact_text(keyword) in normalized for keyword in keywords if keyword)


def load_department_db(department: str):
    """
    รองรับ 2 แบบ:
    1) ถ้ามีแยก DB ตามสาขา -> ใช้อันนั้น
    2) ถ้าไม่มี -> fallback ไปฐานรวม
    """
    department_dir = os.path.join(CHROMA_ROOT, department)
    root_dir = CHROMA_ROOT

    embedding = get_embedding()

    if os.path.exists(department_dir):
        return Chroma(
            persist_directory=department_dir,
            embedding_function=embedding,
            collection_name=f"{department.lower()}_courses",
        )

    if os.path.exists(root_dir):
        return Chroma(
            persist_directory=root_dir,
            embedding_function=embedding,
            collection_name="kmutnb_courses",
        )

    raise FileNotFoundError(f"Department DB not found: {department_dir} or {root_dir}")


def detect_target_role(query: str) -> str:
    q = _normalize_space(query)

    if "data analyst" in q or "นักวิเคราะห์ข้อมูล" in q:
        return "Data Analyst"
    if "bi analyst" in q or "business intelligence" in q:
        return "BI Analyst"
    if "data scientist" in q or "นักวิทยาศาสตร์ข้อมูล" in q:
        return "Data Scientist"
    if "actuary" in q or "นักคณิตศาสตร์ประกันภัย" in q:
        return "Actuary"
    if "risk analyst" in q or "นักวิเคราะห์ความเสี่ยง" in q:
        return "Risk Analyst"
    if "quant" in q or "quantitative analyst" in q:
        return "Quantitative Analyst"
    if "qc" in q or "qa" in q or "quality control" in q or "quality assurance" in q:
        return "QC/QA"
    if "environmental officer" in q or "เจ้าหน้าที่สิ่งแวดล้อม" in q:
        return "Environmental Officer"
    if "cosmetic scientist" in q or "นักวิทยาศาสตร์เครื่องสำอาง" in q:
        return "Cosmetic Scientist"
    if "software engineer" in q or "developer" in q or "programmer" in q or "นักพัฒนาซอฟต์แวร์" in q:
        return "Software Developer"

    return ""


def _normalize_result(doc: Any, score: float) -> Dict[str, Any]:
    md = doc.metadata or {}

    return {
        "page_content": doc.page_content,
        "program": md.get("program", ""),
        "source_major": md.get("source_major", ""),
        "course_code": md.get("course_code", ""),
        "course_name_en": md.get("course_name_en", ""),
        "course_name_th": md.get("course_name_th", ""),
        "category": md.get("category", ""),
        "description": md.get("description", ""),
        "keywords": _to_list(md.get("keywords")),
        "major_focus": _to_list(md.get("major_focus")),
        "skills": _to_list(md.get("skills")),
        "career_tracks": _to_list(md.get("career_tracks")),
        "primary_roles": _to_list(md.get("primary_roles")),
        "intent_tags": _to_list(md.get("intent_tags")),
        "recommended_next": _to_list(md.get("recommended_next")),
        "retrieval_group": md.get("retrieval_group", ""),
        "search_priority": float(md.get("search_priority", 0.0) or 0.0),
        "score": float(score or 0.0),
    }


def same_program(item_program: str, target_program: str) -> bool:
    item_norm = _normalize_space(item_program)
    target_norm = _normalize_space(target_program)

    aliases = {_normalize_space(x) for x in PROGRAM_ALIASES.get(target_program, {target_program})}
    return item_norm in aliases or item_norm == target_norm


def is_query_too_vague(query: str) -> bool:
    q = _compact_text(query)
    if len(q) < 8:
        return True

    vague_only_patterns = [
        "ชอบเรียน",
        "ชอบอะไร",
        "อยากรู้",
        "แนะนำหน่อย",
        "เรียนอะไรดี",
        "ไม่รู้",
        "ยังไม่แน่ใจ",
        "what should i study",
    ]

    return any(pattern in _normalize_space(query) for pattern in vague_only_patterns) and len(q.split()) <= 3


def is_query_mismatched(department: str, query: str) -> bool:
    mismatch_keywords = CROSS_DOMAIN_MISMATCH_KEYWORDS.get(department, set())
    dept_hints = DEPARTMENT_HINT_KEYWORDS.get(department, set())

    has_mismatch = _contains_any(query, mismatch_keywords)
    has_department_hint = _contains_any(query, dept_hints)

    return has_mismatch and not has_department_hint


def should_keep(result: Dict[str, Any], target_role: str) -> bool:
    category = _normalize_space(result.get("category") or "")

    if category == "general":
        return False

    course_name = f"{result.get('course_name_th', '')} {result.get('course_name_en', '')}"
    if not _normalize_space(course_name).strip():
        return False

    return True


def rerank_score(result: Dict[str, Any], target_role: str, query: str) -> float:
    base_score = float(result.get("score", 0.0))
    category = _normalize_space(result.get("category") or "")
    primary_roles = [x.strip() for x in result.get("primary_roles", [])]
    intent_tags = [_normalize_space(x) for x in result.get("intent_tags", [])]
    search_priority = float(result.get("search_priority", 0.0))

    score = base_score * 5.0

    if category == "major_required":
        score += 3.0
    elif category == "core":
        score += 2.5
    elif category == "elective":
        score += 1.8
    elif category == "general":
        score -= 2.0

    if target_role and target_role in primary_roles:
        score += 3.0

    role_to_intents = {
        "Data Analyst": ["data_analyst", "bi_analyst", "data_visualization", "inferential_statistics"],
        "BI Analyst": ["bi_analyst", "data_visualization", "data_analyst"],
        "Data Scientist": ["predictive_analytics", "data_analyst", "inferential_statistics", "machine_learning"],
        "Actuary": ["actuary", "risk_analyst", "financial_mathematics", "insurance_modeling"],
        "Risk Analyst": ["risk_analyst", "business_risk", "financial_mathematics"],
        "Quantitative Analyst": ["quantitative_analyst", "financial_mathematics", "mathematical_modeling"],
        "Environmental Officer": ["environmental_officer", "environmental_analysis"],
        "Cosmetic Scientist": ["cosmetic_scientist", "cosmetic_chemistry", "product_safety"],
        "QC/QA": ["food_qaqc", "health_beauty_qc", "product_safety", "food_safety"],
        "Software Developer": ["software_development", "web_development", "programming", "computer_science"],
    }

    for tag in role_to_intents.get(target_role, []):
        if tag in intent_tags:
            score += 1.2

    query_text = _normalize_space(query)
    joined_text = " ".join([
        result.get("course_name_th", ""),
        result.get("course_name_en", ""),
        result.get("description", ""),
        " ".join(result.get("keywords", [])),
        " ".join(result.get("skills", [])),
        " ".join(result.get("major_focus", [])),
    ]).lower()

    query_tokens = [tok for tok in _compact_text(query_text).split() if len(tok) >= 2]
    overlap = sum(1 for tok in query_tokens if tok in _compact_text(joined_text))
    score += min(overlap * 0.35, 2.1)

    score += search_priority * 2.0
    return round(score, 4)


def build_no_result_payload(reason: str, department: str, query: str) -> List[Dict[str, Any]]:
    return [{
        "course_code": "",
        "course_name_th": "",
        "course_name_en": "",
        "category": "no_result",
        "description": "",
        "keywords": [],
        "major_focus": [],
        "skills": [],
        "career_tracks": [],
        "primary_roles": [],
        "intent_tags": [],
        "recommended_next": [],
        "retrieval_group": "no_result",
        "search_priority": 0.0,
        "score": 0.0,
        "final_score": 0.0,
        "program": DEPARTMENT_PROGRAM_MAP.get(department, department),
        "source_major": DEPARTMENT_PROGRAM_MAP.get(department, department),
        "search_status": reason,
        "suggestion": _build_search_guidance(reason, department, query),
    }]


def _build_search_guidance(reason: str, department: str, query: str) -> str:
    program = DEPARTMENT_PROGRAM_MAP.get(department, department)

    if reason == "query_too_vague":
        return (
            f"คำถามยังกว้างเกินไปสำหรับสาขา {program} "
            f"ควรระบุให้ชัดขึ้นว่าคุณสนใจทักษะ อาชีพ หรือรูปแบบงานแบบใด "
            f"เช่น 'อยากเป็น Frontend Developer ควรเริ่มจากวิชาอะไร' "
            f"หรือ 'ชอบวิเคราะห์ข้อมูลและทำแดชบอร์ด ควรเรียนวิชาไหน'"
        )

    if reason == "query_mismatched_department":
        return (
            f"คำถามนี้ดูไม่สอดคล้องกับสาขา {program} "
            f"ควรเปลี่ยนคำถามให้เกี่ยวกับสายเรียนของสาขานี้โดยตรง "
            f"หรือเปลี่ยนสาขาให้ตรงกับความสนใจของคุณก่อนถามใหม่"
        )

    return (
        f"ไม่พบรายวิชาที่ตรงพอในสาขา {program} "
        f"ควรถามใหม่โดยระบุเป้าหมายอาชีพหรือทักษะให้ชัด เช่น "
        f"'อยากทำ AI', 'อยากเป็น Data Analyst', 'อยากพัฒนาเว็บ', "
        f"'ชอบเขียนโปรแกรมและฐานข้อมูล'"
    )


def search_department(
    department: str,
    query: str,
    original_query: Optional[str] = None,
    k: int = 10
) -> List[Dict[str, Any]]:
    db = load_department_db(department)

    role_query = original_query or query or ""
    target_role = detect_target_role(role_query)
    target_program = DEPARTMENT_PROGRAM_MAP.get(department, department)

    if is_query_too_vague(role_query):
        return build_no_result_payload("query_too_vague", department, role_query)

    if is_query_mismatched(department, role_query):
        return build_no_result_payload("query_mismatched_department", department, role_query)

    raw_k = max(k * 4, 16)

    try:
        docs_with_scores = db.similarity_search_with_relevance_scores(role_query, k=raw_k)
        raw_results = [_normalize_result(doc, score) for doc, score in docs_with_scores]
    except Exception:
        docs = db.similarity_search(role_query, k=raw_k)
        raw_results = [_normalize_result(doc, 0.0) for doc in docs]

    program_filtered = [
        item for item in raw_results
        if same_program(item.get("program") or item.get("source_major") or "", target_program)
    ]

    if not program_filtered:
        return build_no_result_payload("no_program_match", department, role_query)

    filtered = [
        item for item in program_filtered
        if should_keep(item, target_role)
    ]

    if not filtered:
        return build_no_result_payload("no_valid_course_after_filter", department, role_query)

    for item in filtered:
        item["final_score"] = rerank_score(item, target_role, role_query)

    filtered = [
        item for item in filtered
        if item.get("score", 0.0) >= MIN_RELEVANCE_SCORE or item.get("final_score", 0.0) >= MIN_FINAL_SCORE
    ]

    if not filtered:
        return build_no_result_payload("low_relevance", department, role_query)

    deduped: List[Dict[str, Any]] = []
    seen_keys = set()

    for item in sorted(filtered, key=lambda x: x.get("final_score", 0.0), reverse=True):
        key = (
            _normalize_space(item.get("course_code", "")),
            _normalize_space(item.get("course_name_en", "")),
            _normalize_space(item.get("course_name_th", "")),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)

    return deduped[:k]