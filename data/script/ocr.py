import opendataloader_pdf

# Batch all files in one call — each convert() spawns a JVM process, so repeated calls are slow
opendataloader_pdf.convert(
    input_path=["C:\\Users\\luluming\\formal\\BNU\\2026-classcorpus\\data\\raw\\义务教育课程方案.pdf"],
    output_dir="output/",
    format="markdown",
    hybrid="hancom-ai",
    hybrid_hancom_ai_ocr_strategy='force'
)



