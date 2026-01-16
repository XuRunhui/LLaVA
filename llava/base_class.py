import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
import argparse
class ReasonNormalizer:
    """Normalizes reason text to handle variations"""
    def __init__(self):
        # Abbreviation mappings
        self.abbreviations = {
            r'\bpna\b': 'pneumonia', r'\bptx\b': 'pneumothorax', r'\bsob\b': 'shortness of breath',
            r'\bdoe\b': 'dyspnea on exertion', r'\bchf\b': 'congestive heart failure', r'\bams\b': 'altered mental status',
            r'\br/o\b': 'rule out', r'\beval\b': 'evaluate', r'\bs/p\b': 'status post',
            r'\bng\b': 'nasogastric', r'\bet\b': 'endotracheal', r'\bicu\b': 'intensive care unit',
            r'\buti\b': 'urinary tract infection', r'\bcxr\b': 'chest x-ray', r'\bcopd\b': 'chronic obstructive pulmonary disease',
            r'\bmi\b': 'myocardial infarction', r'\bcad\b': 'coronary artery disease', r'\bcp\b': 'chest pain',
            r'\bcabg\b': "coronary artery bypass grafting", r'\bsats\b': 'saturation', r'\bmva\b': 'motor vehicle accident'
        }
    def normalize(self, reason: str) -> str:
        """Apply all normalization steps to a reason string"""
        if not reason:
            return ""
        # Convert to lowercase
        normalized = reason.lower()
        # Remove "History:" prefix
        normalized = re.sub(r'^history:\s*', '', normalized)
        # Remove demographics placeholders
        normalized = re.sub(r'___-year-old\s+(male|female|man|woman|m|f)', '', normalized)
        normalized = re.sub(r'___\s*(year old|y\.?o\.?)\s+(male|female|man|woman|m|f)', '', normalized)
        normalized = re.sub(r'___[mf]\s+with', '', normalized)
        normalized = re.sub(r'\d+-year-old\s+(male|female|man|woman)', '', normalized)

        # Remove leading "with" prefix (merges "chest pain" and "with chest pain")
        normalized = re.sub(r'^\s*with\s+', '', normalized)
        normalized = re.sub(r'^\s*patient\s+with\s+', '', normalized)

        # rexgradient specific
        # Remove explicit sex info like: "sex: male.", "sex:female"
        normalized = re.sub(r'\bsex\s*:\s*(male|female|man|woman|m|f)\b\.?', '', normalized)
        # Remove explicit age info like:
        # "age: 62 years old", "age: 5 months old", "age: 12 days old"
        normalized = re.sub( r'\bage\s*:\s*\d+\s*(years?|months?|days?)\s*old\b\.?', '', normalized )
        # Remove ethnicity info like: "ethnicity: non-hispanic."
        normalized = re.sub( r'\bethnicity\s*:\s*[a-z\-\s]+\b\.?', '', normalized)

        # Remove leading "symptom:" keyword
        normalized = re.sub(r'\bsymptom\s*:\s*', '', normalized)
        normalized = re.sub(r'^\s*with\s+', '', normalized)
        normalized = re.sub(r'^\s*patient\s+with\s+', '', normalized)

        # Expand abbreviations
        for abbrev, full in self.abbreviations.items():
            normalized = re.sub(abbrev, full, normalized)
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove common separators and trailing punctuation
        normalized = re.sub(r'\s*//\s*', ', ', normalized)
        normalized = re.sub(r'[\.;]+$', '', normalized)

        # Clean up extra spaces around punctuation
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = normalized.strip()
        # print(normalized)
        return normalized

    


# class ReasonClassifier:
#     """Classifies reasons into hierarchical categories"""

#     def __init__(self):
#         # Define primary categories with their keywords
#         self.categories = {
#             'respiratory_infection': {
#                 'keywords': [
#                     'pneumonia', 'covid', 'infiltrate', 'pulmonary', 'infection', 'sepsis', 'ppd',
#                     # added (distinct, non-redundant under substring search)
#                     'consolidation', 'airspace', 'opacification', 'ggo', 'ground glass',
#                     'aspiration', 'pneumonitis', 'bronchitis', 'bronchiolitis',
#                     'rsv', 'influenza', 'tb', 'empyema'
#                 ],
#                 'weight': 10
#             },
#             'respiratory_symptoms': {
#                 'keywords': [
#                     'shortness of breath', 'short of breath', 'dyspnea', 'hypoxia', 'hypoxic', 'breath sounds', 'difficulty breathing',
#                     'respiratory distress', 'respiratory failure', 'ards', 'o2 saturation', 'tachypnea', 'asthma',
#                     # added
#                     'sob', 'doe', 'orthopnea', 'pnd', 'desaturation', 'spo2',
#                     'wheeze', 'wheezing', 'stridor', 'rales', 'crackles', 'rhonchi',
#                     'retractions', 'accessory muscle', 'work of breathing',
#                     'copd', 'emphysema', 'bronchospasm',
#                     'cpap', 'bipap', 'high flow', 'hfnc', 'nonrebreather', 'ventilator'
#                 ],
#                 'weight': 8
#             },


#             ########
#             'cough': {
#                 'keywords': [
#                     'cough', 'productive cough',
#                     # added
#                     'dry cough', 'nonproductive', 'sputum', 'phlegm', 'expectoration',
#                     'hemoptysis'
#                 ],
#                 'weight': 7
#             },
#             'chest_pain': {
#                 'keywords': [
#                     'chest pain', 'pleuritic', 'substernal', 'chest_pressure', 'congestion',
#                     # added
#                     'tightness', 'pressure', 'heaviness', 'pleurisy',
#                     'chest wall', 'costochondritis', 'heartburn', 'reflux', 'gerd'
#                 ],
#                 'weight': 9
#             },
#             'cardiac': {
#                 'keywords': [
#                     'congestive heart failure', 'heart failure', 'cardiomegaly', 'arrest',
#                     'cardiopulmonary', 'myocardial infarction', 'coronary artery disease', 'afib',
#                     # added
#                     'chf', 'cad', 'mi', 'stemi', 'nstemi', 'acs', 'ischemia', 'angina',
#                     'arrhythmia', 'vt', 'vf', 'flutter',
#                     'cardiomyopathy', 'pericarditis', 'tamponade'
#                 ],
#                 'weight': 9
#             },
#             'device_placement': {
#                 'keywords': [
#                     'tube', 'line', 'intubated', 'intubation', 'catheter', 'dobbhoff', 'nasogastric',
#                     'endotracheal', 'swan', 'placement', 'position',
#                     # added (avoid phrases containing "tube"/"line" since those are already substrings)
#                     'ett', 'trach', 'picc', 'cvc', 'aline',
#                     'ecmo', 'impella', 'iabp',
#                     'dialysis', 'port', 'malposition', 'kinked', 'coiled', 'dislodged'
#                 ],
#                 'weight': 10
#             },
#             'pneumothorax': {
#                 'keywords': [
#                     'pneumothorax',
#                     # added (avoid anything containing "pneumothorax")
#                     'ptx', 'lucency', 'pleural line'
#                 ],
#                 'weight': 10
#             },
#             'pleural_effusion': {
#                 'keywords': [
#                     'effusion', 'pleural',
#                     # added (avoid anything containing "effusion"/"pleural")
#                     'costophrenic', 'blunting', 'meniscus', 'loculated',
#                     'hemothorax', 'chylothorax'
#                 ],
#                 'weight': 9
#             },
#             'edema': {
#                 'keywords': [
#                     'edema', 'fluid overload',
#                     # added (avoid anything containing "edema" or "fluid overload")
#                     'anasarca', 'third spacing', 'volume overload'
#                 ],
#                 'weight': 8
#             },
#             'fever': {
#                 'keywords': [
#                     'fever', 'febrile', 'temperature',
#                     # added
#                     'pyrexia', 'tmax', 'chills', 'rigors'
#                 ],
#                 'weight': 7
#             },
#             'neurological': {
#                 'keywords': [
#                     'altered mental status', 'confusion', 'syncope', 'unresponsive',
#                     # added
#                     'ams', 'encephalopathy', 'delirium', 'lethargy', 'somnolent', 'obtunded',
#                     'seizure', 'postictal', 'cva', 'tia', 'aphasia', 'dysarthria'
#                 ],
#                 'weight': 8
#             },
#             'trauma': {
#                 'keywords': [
#                     'trauma', 'fall', 'injury', 'accident', 'fracture',
#                     # added
#                     'mvc', 'mva', 'blunt', 'penetrating', 'assault',
#                     'contusion', 'laceration', 'hematoma'
#                 ],
#                 'weight': 9
#             },
#             'monitoring': {
#                 'keywords': [
#                     'interval change', 'follow up', 'comparison', 'assess', 'placement',
#                     'evaluate', 'monitor',
#                     # added
#                     'followup', 'recheck', 'serial', 'trend',
#                     'stable', 'unchanged', 'progression', 'improving', 'worsening',
#                     'repeat imaging', 'surveillance'
#                 ],
#                 'weight': 3
#             },
#             'other_symptoms': {
#                 'keywords': [
#                     'abdominal pain', 'back pain', 'weakness', 'nausea', 'hematemesis',
#                     'vomiting', 'diarrhea', 'consciousness', 'fibrillation', 'stroke', 'seizure',
#                     # added
#                     'fatigue', 'malaise', 'constipation', 'melena', 'hematochezia', 'gi bleed',
#                     'dysuria', 'hematuria', 'headache', 'myalgia', 'palpitations'
#                 ],
#                 'weight': 5
#             },
#             'systemic_conditions': {
#                 'keywords': [
#                     'sepsis',
#                     'leukocytosis',
#                     'hypotension',
#                     'hypoglycemia',
#                     'hyperglycemia',
#                     'smoke inhalation',
#                     'cirrhosis',
#                     'pancreatitis',
#                     'colitis',
#                     'anemia',
#                     # added
#                     'sirs', 'shock', 'vasopressor', 'pressors',
#                     'lactate', 'lactic acidosis', 'metabolic acidosis',
#                     'aki', 'renal failure',
#                     'coagulopathy', 'thrombocytopenia',
#                     'hyponatremia', 'hypernatremia', 'hypokalemia', 'hyperkalemia',
#                     'bacteremia'
#                 ],
#                 'weight': 7
#             },
#             'oncology': {
#                 'keywords': [
#                     'cancer', 'carcinoma', 'tumor', 'mass',
#                     'lymphoma',
#                     'metastasis', 'metastatic',
#                     'hcc',
#                     'renal cell carcinoma',
#                     'lung cancer', 'lung ca',
#                     'melanoma',
#                     'myeloma',
#                     'malignancy',
#                     # added (avoid phrases containing existing substrings like "lung cancer")
#                     'neoplasm', 'nodule', 'lesion', 'adenocarcinoma', 'squamous',
#                     'nsclc', 'sclc', 'mets', 'recurrence'
#                 ],
#                 'weight': 9
#             },
#             'surgical': {
#                 'keywords': [
#                     'post-op', 'postoperative',
#                     'lobectomy',
#                     'thoracotomy',
#                     'thoracocentesis',
#                     'whipple',
#                     'nissen',
#                     'transplant',
#                     'bmt',
#                     'mediastinoscopy',
#                     'vats',
#                     # added
#                     'postop', 's/p', 'status post',
#                     'resection', 'biopsy', 'sternotomy',
#                     'thoracentesis', 'drain'
#                 ],
#                 'weight': 8
#             },
#             'medical_history': {
#                 'keywords': [
#                     'smoking', 'coronary artery bypass grafting', 'thoracentesis', 'motor vehicle', 'fall',
#                     # added
#                     'smoker', 'tobacco', 'pack-year', 'pmh', 'history of',
#                     'htn', 'diabetes', 'ckd', 'copd'
#                 ],
#                 'weight': 6
#             }
#         }

#     def classify(self, normalized_reason: str) -> List[str]:
#         """Classify a normalized reason into categories"""
#         categories_found = []

#         for category, info in self.categories.items():
#             for keyword in info['keywords']:
#                 if keyword in normalized_reason:
#                     categories_found.append((category, info['weight']))
#                     break

#         # Sort by weight (descending) and return category names
#         categories_found.sort(key=lambda x: x[1], reverse=True)
#         return [cat for cat, _ in categories_found]

import re
from typing import List, Dict, Tuple


class ReasonClassifier:
    """
    Classifies normalized reason text into ONE or MORE of six high-level categories:

      1) symptoms_signs
      2) eval_exclusion_followup
      3) conditions_diagnoses
      4) chronic_history_trauma_events
      5) devices_lines_placement
      6) postoperative_procedures

    Notes:
    - This uses substring matching on a normalized, lowercased string (as your current code does).
    - Provide normalized_reason via ReasonNormalizer.normalize().
    """

    def __init__(self):
        # High-level categories with keywords
        # Weights: used only for sorting results (higher = more "specific/important")
        self.categories: Dict[str, Dict[str, object]] = {
            # 1) Symptoms, Chief Complaints, and Physical Signs
            "symptoms_signs": {
                "keywords": [
                    # Pain and discomfort
                    "pain and discomfort",
                    "chest pain",
                    "chest pressure",
                    "heartburn",
                    "pleurisy", "pleuritic",  # "Pleurisy / pleuritic"
                    "epigastric pain",
                    "right upper quadrant", "ruq pain", "ruq",
                    "abdominal pain",

                    # Respiratory symptoms
                    "respiratory symptoms",
                    "shortness of breath", "short of breath",
                    "respiratory distress",
                    "wheezing", "wheeze",
                    "difficulty breathing",
                    "breath sounds",  # user wrote "Breadth sounds" but likely meant breath sounds
                    "dyspnea",
                    "expectoration",
                    "hyperventilation",
                    "cough",
                    "phlegm",
                    "sputum",
                    "sinus congestion",

                    # Systemic and infectious symptoms
                    "fever", "febrile",
                    "chills", "rigors",
                    "low-grade temperature", "low grade temperature",

                    # Oxygenation and respiratory distress
                    "hypoxia", "hypoxic",
                    "desaturation", "desat",
                    "increased oxygen requirement", "increased o2 requirement",
                    "tachypnea",

                    # Lung exam findings
                    "crackles", "rales",
                    "rhonchi",
                    "decreased breath sounds",

                    # Neurologic and constitutional symptoms
                    "weakness",
                    "fatigue",
                    "lightheadedness", "light headedness",
                    "dizziness",
                    "syncope",
                    "balance difficulty",
                    "ataxia",

                    # Mental status changes
                    "mental status changes",
                    "confusion",
                    "altered mental status",
                    "delirium",
                    "somnolence",
                    "unresponsiveness", "unresponsive",

                    # Gastrointestinal and genitourinary symptoms
                    "nausea", "vomiting", "nausea and vomiting",
                    "coffee-ground emesis", "coffee ground emesis",
                    "constipation",
                    "urinary retention",

                    # Swallowing and foreign body sensation
                    "dysphagia",
                    "food impaction",
                    "foreign body sensation",
                    "food bolus",

                    # Bleeding and hematologic symptoms
                    "hemoptysis",
                    "bloody sputum",
                    "epistaxis",

                    # Fluid balance and edema
                    "weight loss",
                    "weight gain",
                    "edema",
                    "facial swelling",

                    # Cardiovascular and visual symptoms
                    "palpitations",
                    "tachycardia",
                    "new-onset hypertension", "new onset hypertension",
                    "visual changes",
                    "anisocoria",

                    # Laboratory or metabolic abnormalities
                    "rash with transaminitis",
                    "anemia",
                    "leukocytosis", "elevated white blood cell count", "elevated wbc",
                    "elevated lactate",
                    "elevated troponin",
                    "hyperglycemia",
                    "diabetic ketoacidosis", "dka",
                    "hypernatremia",
                    "metabolic alkalosis",
                    "increased creatinine", "elevated creatinine",
                ],
                "weight": 7,
            },

            # 2) Evaluation, Exclusion, and Follow-Up Intents
            "eval_exclusion_followup": {
                "keywords": [
                    "evaluate",
                    "reaccumulation of fluid", "re-accumulation of fluid",
                    "comparison to prior", "compare to prior",
                    "follow-up", "follow up", "followup",
                    "baseline",
                    "monitor", "recheck", "repeat imaging", "surveillance",
                ],
                "weight": 5,
            },

            # 3) Medical Conditions and Diagnoses
            "conditions_diagnoses": {
                "keywords": [
                    # Respiratory and thoracic
                    "pneumonia",
                    "covid",
                    "bronchitis", "bronchiectasis",
                    "sarcoidosis",
                    "pulmonary fibrosis",
                    "pulmonary hypertension",
                    "chronic thromboembolic pulmonary hypertension", "cteph",
                    "acute respiratory distress syndrome", "ards",
                    "respiratory failure",
                    "empyema",
                    "aspiration history", "aspiration",
                    "amiodarone toxicity",
                    "drug-induced pneumonitis", "drug induced pneumonitis",
                    "pneumonitis",
                    "methotrexate toxicity",
                    "tb", "tuberculosis",
                    "rsv",

                    # Cardiovascular
                    "congestive heart failure", "chf",
                    "coronary artery disease", "cad",
                    "atrial fibrillation", "afib",
                    "supraventricular tachycardia", "svt",
                    "bradycardia",
                    "heart failure",
                    "right bundle branch block", "rbbb",
                    "nstemi", "stemi",
                    "unstable angina", "angina",
                    "idiopathic cardiomyopathy", "cardiomyopathy",
                    "ischemia",
                    "pulmonary edema",
                    "endocarditis", "mrsa endocarditis",

                    # Neurologic
                    "transient ischemic attack", "tia",
                    "stroke", "cerebrovascular accident", "cva",
                    "middle cerebral artery", "mca infarct",
                    "intracerebral hemorrhage", "ich",
                    "subarachnoid hemorrhage", "sah",
                    "subdural hematoma", "sdh",
                    "intraparenchymal hemorrhage", "iph",
                    "basal ganglia hemorrhage",
                    "seizure",
                    "transient word-finding difficulty", "word finding difficulty",
                    "slurred speech",
                    "facial droop",
                    "hemibody paresis",

                    # Gastrointestinal, hepatic, and renal
                    "cirrhosis", "hcv-related", "hcv related",
                    "primary biliary cirrhosis",
                    "primary sclerosing cholangitis", "psc",
                    "ascites",
                    "hepatic encephalopathy",
                    "alcoholic hepatitis",
                    "gastrointestinal bleeding", "gi bleed",
                    "variceal bleeding",
                    "bright red blood per rectum", "brbpr",
                    "pancreatitis",
                    "concern for cholangitis", "cholangitis",
                    "hyperbilirubinemia",
                    "crohn",
                    "clostridioides difficile", "c difficile", "c. difficile",
                    "small bowel obstruction", "sbo",
                    "chronic kidney disease", "ckd",
                    "chronic renal insufficiency", "cri",
                    "end-stage renal disease", "esrd",
                    "acute renal failure",
                    "rhabdomyolysis",

                    # Infectious and immunologic
                    "septic shock",
                    "sepsis",
                    "bacteremia",
                    "positive blood cultures",
                    "neutropenia",
                    "febrile neutropenia",
                    "epstein–barr virus", "epstein-barr virus", "ebv viremia", "ebv",
                    "natural killer", "nk cell deficiency",
                    "cellulitis",
                    "abscess",
                    "skin and soft tissue infection", "ssti",

                    # Oncology and hematology
                    "lymphoma", "diffuse large b-cell lymphoma", "dlbcl",
                    "hodgkin lymphoma",
                    "waldenström", "waldenstrom macroglobulinemia",
                    "acute myeloid leukemia", "aml",
                    "myelodysplastic syndrome", "mds",
                    "multiple myeloma",
                    "metastatic renal cell carcinoma",
                    "cancer",
                    "melanoma",
                    "kaposi sarcoma",
                    "glioblastoma multiforme", "gbm",
                    "cholangiocarcinoma",
                    "uterine sarcoma",
                    "leiomyosarcoma",
                    "appendiceal carcinoma",
                ],
                "weight": 9,
            },

            # 4) Chronic History, Trauma and Acute Events
            "chronic_history_trauma_events": {
                "keywords": [
                    "fall",
                    "history of", "history:",
                    "smoke", "smoking", "tobacco",
                    "motor vehicle collision", "mvc", "motor vehicle",
                    "pedestrian struck",
                    "assault",
                    "gunshot wound",
                    "polytrauma",
                    "found down",
                    "head injury while on anticoagulation", "on anticoagulation", "warfarin",
                    "injury",
                    "skateboard fall", "missing tooth fragment", "tooth fragment", "aspiration concern",
                ],
                "weight": 8,
            },

            # 5) Medical Devices, Lines, and Placement Verification
            "devices_lines_placement": {
                "keywords": [
                    "endotracheal tube", "ett", "endotracheal",
                    "intubation", "reintubation", "self-extubation", "self extubation",
                    "tracheostomy", "trach mask", "trach",
                    "mechanical ventilation", "ventilation", "ventilator",
                    "nasogastric tube", "ngt", "nasogastric",
                    "dobhoff", "feeding tube", "tip location", "to 40 cm", "40 cm",
                    "percutaneous endoscopic gastrostomy", "peg", "g-tube", "g tube",
                    "picc", "peripherally inserted central catheter",
                    "midline catheter", "midline",
                    "central venous line", "cvl", "central line", "cvc",
                    "internal jugular", "ij line", "left ij", "right ij",
                    "subclavian line", "subclavian",
                    "implanted port", "port-a-cath", "portacath", "port",
                    "non-functioning port", "non functioning port", "port tip location",
                    "swan–ganz", "swan ganz", "swan",
                    "intra-aortic balloon pump", "iabp",
                    "pacemaker", "ppm", "mri compatibility",
                    "implantable cardioverter-defibrillator", "icd", "aicd",
                    "left ventricular assist device", "lvad", "heartmate",
                    "chest tube", "water seal", "air leak",
                    "pigtail catheter", "pigtail", "removal of pigtail",
                    "thoracentesis",
                    "retained capsule",
                    "retained surgical sponge", "intraoperative film",
                    # generic
                    "tube", "line", "placement", "position", "tip",
                ],
                "weight": 10,
            },

            # 6) Postoperative Status and Procedures
            "postoperative_procedures": {
                "keywords": [
                    "coronary artery bypass grafting", "cabg",
                    "aortic valve replacement", "avr",
                    "mitral valve replacement", "mvr",
                    "transcatheter aortic valve replacement", "tavr",
                    "valvuloplasty",
                    "video-assisted thoracoscopic surgery", "vats", "wedge resection",
                    "lobectomy",
                    "thoracotomy",
                    "mediastinoscopy",
                    "whipple procedure", "whipple",
                    "endoscopic retrograde cholangiopancreatography", "ercp",
                    "cholecystectomy",
                    "exploratory laparotomy", "ex-lap", "ex lap",
                    "colectomy",
                    "ileostomy takedown",
                    "liver transplant",
                    "kidney transplant",
                    "partial nephrectomy",
                    "spinal fusion",
                    "laminectomy",
                    "anterior cervical discectomy and fusion", "acdf",
                    "neck hematoma evacuation",
                    "gastropexy",
                    "nissen fundoplication", "nissen",
                    "post-thrombectomy state", "post thrombectomy",
                    "embolectomy evaluation", "embolectomy",
                    "biopsy", "planned brain biopsy", "brain biopsy",
                    "post-op", "postoperative", "postop", "status post", "s/p",
                ],
                "weight": 9,
            },
        }

        # Optional: compile a fast "phrase boundary" regex for each keyword
        # This reduces accidental matches inside longer words (e.g., "port" in "important").
        self._compiled: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        for cat, info in self.categories.items():
            compiled_list = []
            for kw in info["keywords"]:
                # For multi-word phrases, \b works OK at ends; for symbols like "s/p" we skip \b wrapping.
                if any(ch in kw for ch in ["/", "-", "’", "'", "–"]):
                    pat = re.compile(re.escape(kw))
                else:
                    pat = re.compile(r"\b" + re.escape(kw) + r"\b")
                compiled_list.append((kw, pat))
            self._compiled[cat] = compiled_list

    def classify(self, normalized_reason: str) -> List[str]:
        """
        Return matching categories sorted by weight (desc).
        """
        if not normalized_reason:
            return []

        hits: List[Tuple[str, int]] = []
        text = normalized_reason.lower()

        for cat, info in self.categories.items():
            for _, pat in self._compiled[cat]:
                if pat.search(text):
                    hits.append((cat, int(info["weight"])))
                    break

        hits.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in hits]

    def classify_top(self, normalized_reason: str) -> str | None:
        """
        Convenience: return the single best category (highest weight) or None.
        """
        cats = self.classify(normalized_reason)
        return cats[0] if cats else None
