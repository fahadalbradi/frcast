# 🔮 محرك التنبؤ الذكي (Intelligent Prediction Engine)

تنفيذ فعلي لميثاق النظام: محرك Multi-Agent عام (Domain-Agnostic) يأخذ أي بيانات جدولية
(مبيعات، عقارات، عملاء...) ويشغّل الحلقة كاملة تلقائياً:

```
Profiling → Preprocessing → FLAML Training (Self-Correction) → Triple-Threat Evaluation
                                                                  (Statistical / Stability / SHAP)
                                                                → Confidence Score → Predict / LLM Forecast
```

## 📁 هيكل المشروع

```
prediction_engine/
  profiler.py       # المرحلة 1: استخراج بصمة البيانات (Data Fingerprint)
  preprocessor.py    # المرحلة 2: تنظيف وترميز ديناميكي + transform_new() للتنبؤ الحي
  trainer.py          # المرحلة 3: FLAML AutoML مع تحكم بالـ search space / time budget
  evaluator.py         # المرحلة 4: المظلة الثلاثية (R2/MAE، Cross-Validation، SHAP)
  confidence.py         # درجة الثقة (0-1) المبنية على الجودة + الاستقرار + كفاية البيانات
  orchestrator.py        # حلقة ReAct + Self-Correction (إعادة محاولة تلقائية) + Self-Healing
  llm_agent.py             # وكيل DSPy لوضع "LLM Forecast" التحليلي (اختياري، يحتاج مفتاح API)
app.py                      # واجهة Streamlit التفاعلية
requirements.txt
sample_data_real_estate.csv  # بيانات تجريبية (600 عقار، أعمدة مفقودة عمداً لاختبار التنظيف)
```

## 🚀 التشغيل المحلي

```bash
pip install -r requirements.txt
streamlit run app.py
```

سيفتح المتصفح تلقائياً على `http://localhost:8501`.

### تفعيل وضع "LLM Forecast" (اختياري)

الوضع الرقمي **Predict** يعمل دائماً بدون أي مفتاح API (الأرقام تخرج من نموذج FLAML
المدرَّب فعلياً، وليس من الـLLM — تماشياً مع مبدأ الميثاق: Code Interpreter بدل توليد
الأرقام من الـLLM مباشرة).

لتفعيل التحليل النصي التفسيري **LLM Forecast** (يستخدم DSPy + OpenAI):

```bash
export OPENAI_API_KEY="sk-..."
streamlit run app.py
```

أو أدخل المفتاح مباشرة داخل الواجهة، تبويب "LLM Forecast" (لا يُحفظ، يُستخدم للجلسة فقط).

## ✅ ماذا تم اختباره فعلياً (وليس نظرياً)

تم تشغيل المحرك بالكامل على 3 سيناريوهات مختلفة أثناء البناء للتأكد من أنه Domain-Agnostic فعلاً:

1. **انحدار (تسعير عقارات)** — R² ≈ 0.98، اجتاز البوابة النوعية، ثقة "مرتفعة".
2. **بيانات عشوائية بلا إشارة حقيقية** — النظام رفض النتيجة تلقائياً بعد 3 محاولات
   Self-Correction، وأعطى ثقة "منخفضة" وأوصى بعدم الاعتماد على Predict — تماماً
   كما ينص الميثاق ("النظام يرفض النتائج ضعيفة الارتباط ولا يعطي تنبؤات مضللة").
3. **تصنيف (احتمالية مغادرة عميل — دومين مختلف تماماً)** — Accuracy ≈ 0.75، اجتاز
   البوابة، بدون أي كود مخصص لهذا الدومين.

## ⚠️ حدود هذا التنفيذ (شفافية كاملة)

- **الأمان/العزل (Sandbox)**: التنفيذ الحالي يشغّل FLAML/SHAP مباشرة داخل نفس العملية؛
  لبيئة إنتاج حقيقية يُنصح بتشغيل مرحلة Training داخل Sandbox معزول (container منفصل)
  كما ينص الميثاق، خصوصاً إذا كانت البيانات أو الأعمدة تتضمن كوداً أو Formulas.
- **DSPy/ReAct متعدد الخطوات**: `llm_agent.py` يستخدم `dspy.Predict` (خطوة استدلال واحدة)
  لأن وضع LLM Forecast الحالي لا يحتاج أدوات خارجية. لو احتجت الوكيل يستدعي أدوات
  (بحث، استعلام قاعدة بيانات...) يمكن ترقيته لـ `dspy.ReAct` بسهولة لأن البنية جاهزة.
- **Self-Healing**: مُطبّق على مستوى إعادة محاولة التدريب عند الفشل التقني أو ضعف
  الجودة. لم يُطبّق self-healing على مستوى توليد كود Preprocessing ديناميكياً بالكامل
  عبر LLM (التنظيف الحالي قائم على قواعد إحصائية ثابتة وموثوقة بدل كود يولّده LLM في
  كل مرة، لتقليل الأخطاء العشوائية).

## 📊 مصدر الأرقام مقابل التحليل (Predict vs LLM Forecast)

| الوضع | المصدر | الاستخدام |
|---|---|---|
| **Predict** | نموذج FLAML مدرَّب فعلياً (deterministic) | رقم دقيق تعتمد عليه في القرار |
| **LLM Forecast** | OpenAI عبر DSPy | تفسير نصي: لماذا هذا الرقم، ما المخاطر، ما مدى الثقة |
