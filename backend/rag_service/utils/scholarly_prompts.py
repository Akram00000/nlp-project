"""
Scholarly Prompt Templates for Islamic RAG

Provides structured prompts for generating scholarly, provenance-based
answers with multi-view presentation and safety constraints.
"""

# ============================================================
# Confidence Level Definitions
# ============================================================

CONFIDENCE_LEVELS = {
    "definitive": {
        "ar": "قطعي الدلالة",
        "en": "Definitive/Explicit",
        "description": "Direct, explicit textual evidence with unanimous scholarly agreement"
    },
    "presumptive": {
        "ar": "ظني الدلالة",
        "en": "Presumptive/Interpretive",
        "description": "Evidence requiring interpretation; scholars may differ on understanding"
    },
    "disputed": {
        "ar": "خلاف بين العلماء",
        "en": "Scholarly Disagreement",
        "description": "Multiple valid positions exist among qualified scholars"
    },
    "analogical": {
        "ar": "قياس",
        "en": "Analogical Reasoning",
        "description": "Derived through analogy (qiyās) from established rulings"
    },
    "no_direct_evidence": {
        "ar": "لا دليل صريح",
        "en": "No Direct Evidence",
        "description": "No explicit text addresses this specific issue"
    }
}


# ============================================================
# Scholarly System Prompt (Bilingual)
# ============================================================

SCHOLARLY_SYSTEM_PROMPT = """أنت مساعد باحث في الدراسات الإسلامية (الفقيه الذكي). دورك هو تقديم إجابات دقيقة وموثقة مع العزو المناسب للعلماء.

## ⛔ قواعد الاقتباس الصارمة (منع الهلوسة):
❌ لا تذكر "فتوى رقم X" إلا إذا رأيت هذا الرقم تحديداً في نص المصدر الذي تقتبس منه.
❌ لا تخترع أرقام فتاوى أو أحاديث أو صفحات من ذاكرتك.
❌ لا تخمن أسماء الصور أو أرقام الآيات.
❌ لا تستشهد بمصادر غير موجودة في الوثائق المسترجعة.

✅ استخدم دائماً chunk_id للاقتباس بهذا التنسيق حصراً:
   [source:CHUNK_ID]وصف المصدر[/source]
   
✅ استخدم دائماً أوصافاً عامة إذا لم يوجد رقم محدد في النص:
   - "حسب [source:fatwa_0]فتوى من إسلام ويب[/source]"
   - "ورد في [source:hadith_1]الحديث المذكور[/source]"

## 📜 هيكل الإجابة الإلزامي (التزم به تماماً):
يمنع إضافة أي عناوين أو خواتم غير المذكورة أدناه. الإجابة يجب أن تكون باللغة العربية الفصحى فقط.

### المستوى 1 - الملخص
ملخص من سطرين أو ثلاثة للحكم الأساسي.

### المستوى 2 - الحكم الشرعي
ذكر الحكم بوضوح مع تصنيفه (إجماع / قول الجمهور / قول بعض العلماء).

### المستوى 3 - التفصيل والأدلة
شرح مفصل مع أدلة كاملة من المصادر المسترجعة.

### المستوى 4 - الآراء الفقهية
عرض الآراء المختلفة مع نسبتها لأصحابها.

<details>
<summary>📖 التحليل المعمق</summary>
للخبراء: الشروط، الاستثناءات، البدائل، تفاصيل الاستدلال.
</details>

## ⚠️ تنبيه خاص (القروض السكنية والربا):
عند السؤال عن شراء المنازل بالقروض في الغرب، يجب تضمين هذه المعلومات في "المستوى 4":
- ذكر رأي المجلس الأوروبي للإفتاء (إجازة شراء المسكن بشرط ألا يملك غيره وللسكن الشخصي وعدم وجود بديل).
- ذكر رد الجمهور (أن الربا محرم مطلقاً وأن الحاجة لا تبيح الربا).

❌ لا تقل أبداً "قرض ربوي بدون فائدة" - هذا تناقض منطقي!
❌ لا تدرج هذه التعليمات أو القواعد في إجابتك النهائية. ابدأ مباشرة بالمستوى 1.

- ابدأ بكتلة <thought> تظهر فيها تحليلك للمصادر (سيتم حذفها تلقائياً).
"""


# ============================================================
# User Prompt Template
# ============================================================

def build_scholarly_user_prompt(query: str, context: str, language: str = "ar") -> str:
    """Build the user prompt with context for scholarly generation."""
    
    if language == "ar":
        return f"""السؤال:
{query}

المصادر المتاحة للاستدلال:
{context}

أجب وفق الصيغة المحددة. إذا لم تجد دليلاً صريحاً في المصادر، صرّح بذلك."""
    else:
        return f"""Question:
{query}

Available sources for evidence:
{context}

Answer according to the specified format. If no explicit evidence exists in the sources, state that clearly."""


# ============================================================
# Validation Patterns
# ============================================================

import re

QURAN_REFERENCE_PATTERN = re.compile(
    r'(?:سورة\s+)?(\w+)\s*[:\-]\s*(\d+)(?:\s*-\s*(\d+))?|'
    r'(?:Surah\s+)?(\w+)\s*[:\-]\s*(\d+)(?:\s*-\s*(\d+))?',
    re.UNICODE
)

HADITH_REFERENCE_PATTERN = re.compile(
    r'(?:صحيح\s+)?(?:البخاري|مسلم|الترمذي|أبو داود|النسائي|ابن ماجه)\s*[:\-#]\s*(\d+)|'
    r'(?:Sahih\s+)?(?:Bukhari|Muslim|Tirmidhi|Abu Dawud|Nasai|Ibn Majah)\s*[:\-#]\s*(\d+)',
    re.UNICODE | re.IGNORECASE
)


def validate_references(text: str) -> dict:
    """
    Validate references in generated text.
    
    Returns dict with found references and any potential issues.
    """
    quran_refs = QURAN_REFERENCE_PATTERN.findall(text)
    hadith_refs = HADITH_REFERENCE_PATTERN.findall(text)
    
    return {
        "quran_references": len(quran_refs),
        "hadith_references": len(hadith_refs),
        "has_references": len(quran_refs) > 0 or len(hadith_refs) > 0,
        "raw_quran": quran_refs,
        "raw_hadith": hadith_refs
    }


def check_no_evidence_statement(text: str) -> bool:
    """Check if the text properly states when no evidence was found."""
    no_evidence_markers = [
        "لم يتم العثور",
        "لا دليل صريح",
        "لم أجد في المصادر",
        "no explicit evidence",
        "no direct evidence",
        "not found in the sources"
    ]
    return any(marker in text.lower() for marker in no_evidence_markers)
