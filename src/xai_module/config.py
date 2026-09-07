"""
Centralized Configuration for XAI Module
==========================================

All configurable thresholds, rules, and constants are defined here in one place.
Modify this file to adjust severity thresholds, treatment rules, forecasting
parameters, or any other tunable settings.

NOTE: Severity thresholds and treatment rules are project-defined defaults.
They are NOT medically or agriculturally validated. Consult domain experts
before deploying in a real agricultural setting.
"""

# ==============================================================================
# Severity Level Thresholds
# ==============================================================================
# Maps severity percentage ranges to human-readable categories.
# Format: list of (upper_bound_exclusive, label) sorted ascending.
# The last entry covers everything up to 100%.
SEVERITY_THRESHOLDS = [
    (10.0, "Low"),
    (30.0, "Mild"),
    (60.0, "Moderate"),
    (100.01, "Severe"),  # 100.01 to include exactly 100%
]


# ==============================================================================
# Forecasting Defaults
# ==============================================================================
FORECAST_MIN_DATA_POINTS = 2        # Minimum observations to attempt forecast
FORECAST_DEFAULT_FUTURE_DAYS = 7    # Default number of days to forecast ahead
FORECAST_DEFAULT_METHOD = "linear"  # "linear" or "exponential"


# ==============================================================================
# Risk Level Mapping
# ==============================================================================
# Based on current severity level + trend direction → risk label.
# Format: (severity_level, trend) → risk_level
RISK_LEVEL_MAP = {
    ("Low", "Increasing"):       "Moderate",
    ("Low", "Stable"):           "Low",
    ("Low", "Decreasing"):       "Low",
    ("Mild", "Increasing"):      "Moderate",
    ("Mild", "Stable"):          "Low",
    ("Mild", "Decreasing"):      "Low",
    ("Moderate", "Increasing"):  "High",
    ("Moderate", "Stable"):      "Moderate",
    ("Moderate", "Decreasing"):  "Moderate",
    ("Severe", "Increasing"):    "Critical",
    ("Severe", "Stable"):        "High",
    ("Severe", "Decreasing"):    "High",
}


# ==============================================================================
# Treatment Recommendation Rules
# ==============================================================================
# Disease-specific high-level recommendations.
# Keys use the PlantVillage naming convention: "Crop___Disease"
#
# Each entry contains recommendations keyed by severity level.
# If a disease is not found here, the GENERIC_RECOMMENDATIONS fallback is used.
#
# IMPORTANT: These are general agricultural best-practice suggestions ONLY.
# No specific chemical dosages or unsafe pesticide instructions are included.
TREATMENT_RULES = {
    "Tomato___Late_blight": {
        "Low": (
            "Monitor plants closely for spreading lesions. "
            "Remove any affected leaves and dispose of them away from the field. "
            "Ensure adequate plant spacing for air circulation."
        ),
        "Mild": (
            "Remove and destroy infected plant parts. "
            "Apply approved copper-based fungicide per local agricultural guidelines. "
            "Avoid overhead irrigation to reduce leaf wetness."
        ),
        "Moderate": (
            "Isolate severely affected plants immediately. "
            "Apply locally approved systemic fungicide as per label instructions. "
            "Consider removing heavily infected plants to prevent further spread. "
            "Consult a local agricultural extension officer."
        ),
        "Severe": (
            "Remove and destroy all heavily infected plants. "
            "Apply emergency fungicide treatment following local regulations. "
            "Implement crop rotation for future planting seasons. "
            "Seek immediate consultation with an agricultural expert."
        ),
    },
    "Tomato___Early_blight": {
        "Low": (
            "Remove lower infected leaves. Mulch around the base of plants "
            "to prevent soil splash. Maintain proper nutrition."
        ),
        "Mild": (
            "Apply approved protectant fungicide. Remove infected debris. "
            "Ensure proper plant spacing and staking for airflow."
        ),
        "Moderate": (
            "Apply systemic fungicide per local guidelines. "
            "Remove heavily infected plant material. "
            "Improve field drainage and avoid working with wet plants."
        ),
        "Severe": (
            "Remove and destroy severely infected plants. "
            "Apply approved fungicide treatment urgently. "
            "Plan crop rotation for next season. Consult an agronomist."
        ),
    },
    "Tomato___Bacterial_spot": {
        "Low": "Remove affected leaves. Avoid overhead watering. Monitor spread.",
        "Mild": (
            "Apply copper-based bactericide per label instructions. "
            "Use disease-free seeds and transplants for future planting."
        ),
        "Moderate": (
            "Isolate affected plants. Apply approved treatments. "
            "Sanitize tools between plants. Consult a plant pathologist."
        ),
        "Severe": (
            "Remove and destroy infected plants. Treat remaining plants preventively. "
            "Disinfect all tools and equipment. Seek expert advice for replanting."
        ),
    },
    "Potato___Late_blight": {
        "Low": "Scout fields regularly. Remove volunteer potato plants. Monitor weather conditions.",
        "Mild": (
            "Apply protectant fungicide. Hill soil around stems. "
            "Avoid irrigation before forecasted wet periods."
        ),
        "Moderate": (
            "Apply systemic fungicide per local agricultural extension recommendations. "
            "Remove heavily infected plants and destroy them. "
            "Do not compost infected material."
        ),
        "Severe": (
            "Destroy all infected foliage (vine-kill). "
            "Wait before harvesting to prevent tuber infection. "
            "Implement strict crop rotation. Consult an expert immediately."
        ),
    },
    "Potato___Early_blight": {
        "Low": "Maintain plant vigor with adequate fertilization. Remove lower infected leaves.",
        "Mild": "Apply approved fungicide. Improve air circulation. Avoid wetting foliage.",
        "Moderate": (
            "Apply systemic fungicide. Remove significant infected material. "
            "Adjust irrigation schedule."
        ),
        "Severe": (
            "Consider early harvest if tubers are mature enough. "
            "Destroy infected vines. Plan rotation and resistant varieties."
        ),
    },
    "Apple___Apple_scab": {
        "Low": "Rake and destroy fallen leaves. Prune for better air circulation.",
        "Mild": "Apply approved fungicide at green tip stage. Continue sanitation practices.",
        "Moderate": (
            "Apply fungicide program through the season. "
            "Remove heavily infected fruit and leaves."
        ),
        "Severe": (
            "Implement intensive fungicide program per local recommendations. "
            "Consider replacing with scab-resistant varieties. Consult an arborist."
        ),
    },
    "Apple___Black_rot": {
        "Low": "Remove mummified fruit and prune dead wood. Maintain tree vigor.",
        "Mild": "Apply fungicide during early season. Remove cankers from branches.",
        "Moderate": (
            "Intensify fungicide applications. Remove all infected fruit and wood. "
            "Improve orchard sanitation."
        ),
        "Severe": (
            "Prune severely infected branches. Apply fungicide aggressively. "
            "Consult a certified arborist or extension agent."
        ),
    },
    "Grape___Black_rot": {
        "Low": "Remove mummified berries and infected leaves. Maintain canopy airflow.",
        "Mild": "Apply approved fungicide before bloom. Practice good vineyard sanitation.",
        "Moderate": (
            "Apply fungicide program from bud break through veraison. "
            "Remove infected clusters promptly."
        ),
        "Severe": (
            "Intensive fungicide program required. Remove all infected material. "
            "Consult a viticulture specialist for long-term management."
        ),
    },
    "Corn_(maize)___Common_rust_": {
        "Low": "Monitor rust pustule development. No action typically needed for light infections.",
        "Mild": "Apply foliar fungicide if economic threshold is reached. Scout regularly.",
        "Moderate": (
            "Apply approved fungicide. Consider planting resistant hybrids next season."
        ),
        "Severe": (
            "Apply fungicide immediately. Assess yield loss potential. "
            "Plan resistant hybrid selection for future planting."
        ),
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Low": "Monitor lesion progression. Maintain plant health.",
        "Mild": "Apply foliar fungicide at early tassel stage if warranted.",
        "Moderate": (
            "Apply fungicide. Remove crop debris post-harvest. "
            "Consider resistant hybrids."
        ),
        "Severe": (
            "Apply fungicide urgently. Plan crop rotation and tillage "
            "to reduce inoculum. Seek agronomist consultation."
        ),
    },
}


# ==============================================================================
# Generic Fallback Recommendations (when disease-specific rules are not available)
# ==============================================================================
GENERIC_RECOMMENDATIONS = {
    "Low": (
        "Continue regular monitoring. Remove any visibly affected leaves. "
        "Maintain good field hygiene and proper plant nutrition."
    ),
    "Mild": (
        "Increase monitoring frequency. Remove infected plant material. "
        "Improve air circulation around plants. "
        "Consider applying locally approved preventive treatments."
    ),
    "Moderate": (
        "Isolate affected plants where possible. "
        "Apply locally approved disease management treatments. "
        "Remove and safely dispose of heavily infected material. "
        "Consult a local agricultural extension officer for guidance."
    ),
    "Severe": (
        "Immediate intervention required. Remove and destroy heavily infected plants. "
        "Apply approved treatments following local regulations. "
        "Implement crop rotation planning. "
        "Seek urgent consultation with an agricultural expert or plant pathologist."
    ),
}


# ==============================================================================
# Monitoring Recommendations (based on severity level)
# ==============================================================================
MONITORING_RECOMMENDATIONS = {
    "Low": "Inspect plants weekly. Photograph affected areas for comparison over time.",
    "Mild": "Inspect plants every 3-4 days. Track spread to neighboring plants.",
    "Moderate": "Inspect plants daily. Monitor weather conditions favorable to disease spread.",
    "Severe": "Inspect twice daily. Monitor all nearby plants and fields for spread.",
}


# ==============================================================================
# Progression-Based Action Modifiers
# ==============================================================================
# Additional advice appended when disease is progressing rapidly.
PROGRESSION_MODIFIERS = {
    "Increasing": " Disease is actively spreading — act promptly to prevent further damage.",
    "Stable": " Disease appears stable — maintain current management and continue monitoring.",
    "Decreasing": " Disease appears to be subsiding — continue monitoring and maintain preventive measures.",
}


# ==============================================================================
# Standard Disclaimer
# ==============================================================================
DISCLAIMER = (
    "AI-generated recommendation for decision support only. "
    "Consult a qualified agricultural expert or local extension service "
    "before applying any treatment. Do not rely solely on this system "
    "for critical crop management decisions."
)


# ==============================================================================
# Output Paths
# ==============================================================================
DEFAULT_OUTPUT_DIR = "outputs"
