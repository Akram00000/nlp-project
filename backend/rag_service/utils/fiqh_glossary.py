"""
Fiqh Glossary Module - Islamic Legal Term Definitions

Provides a glossary of common fiqh terms with Arabic definitions,
and utilities to inject inline definitions into generated text.
"""

from typing import Dict, Tuple
import re


# ============================================================
# Fiqh Terms Glossary
# ============================================================

FIQH_GLOSSARY: Dict[str, Dict[str, str]] = {
    # Fundamental Legal Categories
    "ضرورة": {
        "ar": "الحالة التي يخشى فيها على النفس أو العرض أو المال إن لم يفعل المحرم، كالخوف من الموت أو الضرر البالغ",
        "en": "Necessity - a situation where one fears for life, honor, or wealth if the prohibited act is not done"
    },
    "حاجة": {
        "ar": "ما يحتاج إليه المرء لدفع المشقة الشديدة دون الوصول لحد الضرورة",
        "en": "Need - what one requires to remove severe hardship without reaching the level of necessity"
    },
    "مباح": {
        "ar": "ما لا ثواب في فعله ولا عقاب في تركه",
        "en": "Permissible - an act with no reward for doing it and no punishment for leaving it"
    },
    "محرم": {
        "ar": "ما نهى عنه الشارع نهياً جازماً، يُثاب تاركه ويُعاقب فاعله",
        "en": "Prohibited - what the Lawgiver has strictly forbidden; one is rewarded for avoiding it"
    },
    "واجب": {
        "ar": "ما أمر به الشارع أمراً جازماً، يُثاب فاعله ويُعاقب تاركه",
        "en": "Obligatory - what the Lawgiver has strictly commanded; one is rewarded for doing it"
    },
    "مندوب": {
        "ar": "ما أمر به الشارع لا على وجه الإلزام، يُثاب فاعله ولا يُعاقب تاركه",
        "en": "Recommended - commanded without obligation; rewarded for doing, not punished for leaving"
    },
    "مكروه": {
        "ar": "ما نهى عنه الشارع لا على وجه الإلزام، يُثاب تاركه ولا يُعاقب فاعله",
        "en": "Disliked - discouraged without prohibition; rewarded for avoiding, not punished for doing"
    },
    
    # Transaction Terms
    "ربا": {
        "ar": "الزيادة في أحد البدلين المتجانسين من غير أن تقابلها زيادة في البدل الآخر",
        "en": "Usury/Interest - an increase in one of two exchanged items without a corresponding increase"
    },
    "غرر": {
        "ar": "الجهالة في المبيع أو الثمن أو الأجل التي قد تؤدي إلى النزاع",
        "en": "Uncertainty - ambiguity in the sale item, price, or term that may lead to dispute"
    },
    "مرابحة": {
        "ar": "بيع السلعة بثمنها الأصلي مع ربح معلوم متفق عليه",
        "en": "Cost-plus sale - selling an item at its original price plus an agreed-upon profit"
    },
    
    # Scholarly Consensus Terms
    "إجماع": {
        "ar": "اتفاق مجتهدي الأمة في عصر على حكم شرعي",
        "en": "Consensus - agreement of the scholars of the Ummah in an era on a legal ruling"
    },
    "قياس": {
        "ar": "إلحاق فرع بأصل لعلة جامعة بينهما في الحكم",
        "en": "Analogical reasoning - applying a ruling to a new case based on a shared cause"
    },
    "اجتهاد": {
        "ar": "بذل الوسع في استنباط الأحكام الشرعية من أدلتها التفصيلية",
        "en": "Independent reasoning - exerting effort to derive rulings from detailed evidence"
    },
    "تقليد": {
        "ar": "اتباع قول المجتهد من غير معرفة دليله",
        "en": "Following - adhering to a scholar's opinion without knowing its evidence"
    },
    
    # Worship Terms
    "فرض عين": {
        "ar": "ما يجب على كل مكلف بعينه كالصلوات الخمس",
        "en": "Individual obligation - required of every accountable person, like the five prayers"
    },
    "فرض كفاية": {
        "ar": "ما يجب على مجموع المكلفين لا على كل فرد، فإذا قام به البعض سقط عن الباقين",
        "en": "Collective obligation - if some fulfill it, the obligation is lifted from others"
    },
    "سنة مؤكدة": {
        "ar": "ما واظب عليه النبي ﷺ ولم يتركه إلا نادراً",
        "en": "Emphasized Sunnah - what the Prophet ﷺ consistently practiced, rarely leaving it"
    },
}


# ============================================================
# Consensus Level Labels
# ============================================================

CONSENSUS_LABELS = {
    "ijma": {
        "ar": "الإجماع",
        "en": "Consensus",
        "description": "All major madhahib agree on this ruling",
        "css_class": "consensus-ijma"
    },
    "jumhur": {
        "ar": "قول الجمهور",
        "en": "Majority Opinion",
        "description": "Position held by 3 or more madhahib",
        "css_class": "consensus-jumhur"
    },
    "ba3d": {
        "ar": "قول بعض العلماء",
        "en": "Some Scholars' View",
        "description": "Minority position held by respected scholars",
        "css_class": "consensus-minority"
    },
    "shadh": {
        "ar": "رأي شاذ",
        "en": "Irregular Opinion",
        "description": "Clearly outlier position not widely accepted",
        "css_class": "consensus-outlier"
    },
}


# ============================================================
# Text Processing Functions
# ============================================================

def inject_glossary_definitions(text: str, format: str = "tooltip") -> str:
    """
    Inject glossary term markers into text for frontend rendering.
    
    Args:
        text: The text to process
        format: 'tooltip' for {{term}} markers, 'footnote' for [^term] markers
        
    Returns:
        Text with term markers injected
    """
    processed_text = text
    
    for term in FIQH_GLOSSARY.keys():
        # Match the term with word boundaries (Arabic-aware)
        # Look for standalone terms, not already marked
        pattern = rf'(?<!\{{)\b{re.escape(term)}\b(?!\}})'
        
        if format == "tooltip":
            replacement = f'{{{{{term}}}}}'  # {{term}}
        else:
            replacement = f'{term}[^{term}]'
            
        # Only replace first occurrence to avoid cluttering
        processed_text = re.sub(pattern, replacement, processed_text, count=1)
    
    return processed_text


def get_term_definition(term: str, lang: str = "ar") -> str:
    """Get the definition of a fiqh term."""
    if term in FIQH_GLOSSARY:
        return FIQH_GLOSSARY[term].get(lang, FIQH_GLOSSARY[term]["ar"])
    return ""


def build_glossary_footnotes(text: str) -> str:
    """
    Build footnote definitions for terms used in text.
    
    Returns markdown-style footnotes to append to the answer.
    """
    footnotes = []
    
    for term, definitions in FIQH_GLOSSARY.items():
        if term in text:
            footnotes.append(f"[^{term}]: **{term}**: {definitions['ar']}")
    
    if footnotes:
        return "\n\n---\n### مصطلحات / Glossary\n" + "\n".join(footnotes)
    return ""


def format_consensus_label(level: str, lang: str = "ar") -> str:
    """
    Format a consensus level as a styled label.
    
    Args:
        level: One of 'ijma', 'jumhur', 'ba3d', 'shadh'
        lang: Language for label text
        
    Returns:
        Markdown-formatted label with CSS class marker
    """
    if level not in CONSENSUS_LABELS:
        return ""
    
    label_info = CONSENSUS_LABELS[level]
    label_text = label_info[lang]
    css_class = label_info["css_class"]
    
    # Return markdown with custom marker for frontend parsing
    return f'<span class="{css_class}">**{label_text}**</span>'


def format_minority_opinion_box(
    opinion_holder: str,
    opinion_text: str,
    rebuttal_text: str = "",
    lang: str = "ar"
) -> str:
    """
    Format a minority opinion with optional rebuttal.
    
    Returns a styled warning box in markdown format.
    """
    if lang == "ar":
        header = f"⚠️ رأي {opinion_holder} (Minority Opinion)"
        rebuttal_header = "**الرد على هذا الرأي** (Rebuttal by Majority):"
    else:
        header = f"⚠️ Opinion of {opinion_holder} (Minority)"
        rebuttal_header = "**Rebuttal by Majority:**"
    
    box = f"""
<div class="minority-opinion-box">

### {header}

{opinion_text}

"""
    if rebuttal_text:
        box += f"""
{rebuttal_header}

{rebuttal_text}

"""
    box += "</div>"
    
    return box
