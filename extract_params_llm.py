from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import qn
import re
import torch
import os


# ----------------------- docx  to  prompt ----------------------
def iter_block_items(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, doc)
        elif child.tag == qn('w:tbl'):
            yield Table(child, doc)

def table_to_markdown(table):
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
        rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    sep = ['---'] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)

def docx_to_markdown_text(filepath):
    doc = Document(filepath)
    parts = []
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                parts.append(text)
        elif isinstance(block, Table):
            parts.append("[表格开始]\n" + table_to_markdown(block) + "\n[表格结束]")
    return "\n".join(parts)  # 用单个换行连接，避免空白行

def remove_section(text):
    def extract_between_markers(text, start_marker, end_marker, occurrence=2):
        """提取从第 occurrence 次出现 start_marker 到后续第一个 end_marker 之间的内容"""
        starts = [m.start() for m in re.finditer(re.escape(start_marker), text)]
        if len(starts) < occurrence:
            return ""
        start_index = starts[occurrence - 1]
        end_match = re.search(re.escape(end_marker), text[start_index:])
        if not end_match:
            return ""
        end_index = start_index + end_match.start()
        return text[start_index:end_index]
    part0 = extract_between_markers(text, "区块", "设计单位", occurrence=1)
    part1 = extract_between_markers(text, "1-3", "1.4", occurrence=1)
    part2 = extract_between_markers(text, "施工工序及要求", "泵注程序", occurrence=2)
    return part0 + "\n" + part1 + "\n" + part2


def docx_to_model_prompt(filepath):
    """
    返回清洗后适合大模型直接读取的文本 prompt 字符串。
    """
    text = docx_to_markdown_text(filepath)
    cleaned_text = remove_section(text)
    return cleaned_text
#-------------------------------------------------------------------------------


# ----------------------------- LLM inference---------------------------------
def qwen(user_input: str, tokenizer, model, system_prompt: str = "你是一个大语言模型。") -> str:
    """
    给定用户输入，使用 Qwen Instruct 模型生成一次响应。
    参数：
    - user_input: 用户输入的文本
    - tokenizer: 已加载的分词器
    - model: 已加载的模型
    - system_prompt: 可选的system角色设定

    返回：
    - 模型生成的回答文本（不包含提示内容）
    """
    # 构造单轮对话
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    # 转换为 prompt 文本
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 编码为模型输入
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    torch.cuda.empty_cache()
    # 推理生成输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=650,
            temperature=0.1, 
        )
    # 截取模型生成的回答部分（去掉prompt）
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return response
# -----------------------------------------------------------------------------------

def qwen_multi_response(guide, tokenizer, model, system_prompt=None, num_responses=3):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": guide})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        num_return_sequences=num_responses  # 生成多个序列
    )

    responses = [
        tokenizer.decode(generated_ids[i][len(model_inputs["input_ids"][0]):], skip_special_tokens=True)
        for i in range(num_responses)
    ]
    return responses






from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import json
import os
import pandas as pd
import random


def CoT_and_ICL(dataset):
    if dataset == 'camelyon_all':
        normal_1 = ['Uniform cell size: Most lymphocytes are small and similar in size', 
                    'Round or oval nuclei: Nuclei are generally round or slightly oval with clear boundaries', 
                    'Consistent staining: Nuclei stain deep purple (basophilic), and cytoplasm is light pink', 
                    'High nuclear-to-cytoplasm ratio: Nuclei are large relative to cytoplasm, but consistent across cells', 
                    'Even nuclear chromatin: Nuclear staining is homogeneous without coarse granularity or heterogeneity', 
                    'Regular nuclear distribution: Cells are densely packed but organized, without crowding or clustering', 
                    'Absence of nuclear atypia: No multinucleation, irregular nuclear shapes, or abnormal mitotic figures',
                    'Clear cell boundaries: Adjacent cells are well-demarcated, without fusion or overlap',
                    'No atypical cell clusters: No groups of morphologically abnormal cells are present',
                    'Intact tissue architecture: Follicles, cortex, and medulla are clearly defined, and cellular arrangement follows normal physiological patterns']
        normal_2 = ['Intact capsule: a well-defined connective tissue boundary surrounding the lymph node',
                    'Clear cortex structure: dense cortical regions with uniformly arranged lymphoid follicles',
                    'Normal medulla: orderly medullary cords and open sinuses without disruptive lesions', 
                    'Regular follicles: round or oval in shape, with pale germinal centers in the middle', 
                    'Open lymphatic sinuses: sinusoids are patent, containing scattered lymphocytes or macrophages but no dense infiltrates',
                    'Normal vascular distribution: capillaries and small vessels are regularly arranged, without abnormal dilation or neovascularization',
                    'Clear interfollicular regions: uniform stromal fibers without fibrosis or necrosis',
                    'Even lymphocyte density: lymphocytes are evenly distributed in both cortex and medulla, with no abnormal clustering or voids',
                    'Absence of abnormal cell clusters: no atypical cells, nuclear enlargement, or necrotic foci',
                    'Uniform staining: consistent H&E coloration, with pink cytoplasm and dark purple nuclei, and no obvious blotches or unstained areas']
        normal_3 = ['Nuclei stained blue-purple: Hematoxylin imparts a uniform blue-purple color to cell nuclei with clear edges',
                    'Cytoplasm stained pink: Eosin colors most cytoplasm in light pink or pale red',
                    'Clear tissue structures: Distinct demarcation of follicles, sinusoids, and lymphocyte-rich versus stromal areas', 
                    'Uniform color distribution: Regions of the same tissue type show consistent staining intensity without blotches', 
                    'No abnormal dark red or deep purple regions: Absence of excessive blood, necrosis, or hemorrhage',
                    'Connective tissue lightly stained: Collagen and fibrous tissue appear in pale pink or light blue',
                    'Normal blood cell color in vessels: Red blood cells appear orange-red, and vessel outlines are clear',
                    'Lightly stained follicle centers: Germinal centers and follicle cores appear pale pink or light blue',
                    'Even coloration in lymphocyte-rich areas: Dense lymphocyte zones are uniformly blue-purple without localized dark spots',
                    'No non-physiological speckles: Absence of pigment deposits, debris, or staining artifacts']
        normal_4 = ['Uniform follicle distribution: Lymphoid follicles are consistent in size, orderly arranged, with clear boundaries',
                    'Regular sinus structures: Lymphatic sinuses form a mesh-like or radiating pattern, maintaining normal patency without abnormal dilation',
                    'Even cell density: Lymphocytes are distributed uniformly, without localized overcrowding or sparsity', 
                    'Consistent staining intensity: H&E staining is uniform; nuclei appear blue-purple and cytoplasm pink, with no obvious color blotches', 
                    'Ordered cortex and medulla architecture: Cortical, medullary, and marginal regions are clearly delineated and layered',
                    'Smooth boundary transitions: Transitions between tissue regions are natural, without abrupt changes or protrusions',
                    'Regular vascular and small duct orientation: Blood vessels and small ducts are slender, orderly, and follow natural paths without abnormal dilation or breaks',
                    'Absence of abnormal masses or cavities: Tissue textures are continuous, without noticeable voids or nodular formations',
                    'Stable texture orientation: Fibrous connective tissue and lymphocyte arrangements follow natural orientation without abrupt directional changes',
                    'Low local noise: Cell distribution and tissue texture lack isolated artifacts or abnormally bright spots']
        
        tumor_1 = ['Nuclear enlargement: tumor cells often have larger nuclei compared to normal lymphocytes', 
                   'Irregular nuclear shape: nuclei may appear pleomorphic or misshapen',
                   'Hyperchromasia: nuclei show increased staining intensity due to dense chromatin',
                   'High nuclear-to-cytoplasmic ratio: the nucleus occupies a larger portion of the cell',
                   'Prominent nucleoli: visible, enlarged nucleoli within nuclei',
                   'Increased mitotic figures: higher number of cells undergoing division',
                   'Loss of normal cell polarity: disorganized orientation of cells within tissue',
                   'Cellular crowding: densely packed cells, often forming clusters or sheets',
                   'Anisocytosis: variability in cell size across the tissue',
                   'Nuclear membrane irregularities: irregular or indented nuclear borders']
        tumor_2 = ['Disrupted lymph node capsule: the outer boundary of the lymph node is breached by tumor cells', 
                   'Loss of normal follicular structure: lymphoid follicles appear distorted or absent',
                   'Sinusoidal dilation or obstruction: lymphatic sinuses are enlarged or filled with tumor cells',
                   'Replacement of normal parenchyma: normal lymph node tissue is replaced by sheets of tumor cells',
                   'Infiltrative growth patterns: tumor cells infiltrate surrounding stromal tissue irregularly.',
                   'Heterogeneous tissue density: uneven distribution of cellular and extracellular regions',
                   'Formation of micrometastatic clusters: small aggregates of tumor cells scattered throughout the node',
                   'Necrotic areas within tissue: localized cell death due to tumor proliferation',
                   'Fibrotic or desmoplastic response: abnormal connective tissue growth around tumor regions',
                   'Irregular or distorted vascular structures: blood vessels within the lymph node are compressed, dilated, or irregularly branched']
        tumor_3 = ['Increased nuclear basophilia: darker purple nuclei due to higher DNA content in tumor cells', 
                   'Hyperchromatic nuclei: nuclei appear more intensely stained than in normal tissue',
                   'Pale or eosinophilic cytoplasm: abnormal cells may show lighter pink cytoplasm compared to surrounding lymphocytes',
                   'Uneven H&E staining: irregular staining intensity across tumor regions',
                   'Prominent nucleoli: nucleoli stained intensely, appearing as bright spots within nuclei',
                   'High nucleus-to-cytoplasm (N/C) ratio: more purple nuclear staining relative to cytoplasm',
                   'Diffuse or patchy staining patterns: loss of normal tissue color homogeneity',
                   'Presence of necrotic areas: pale or ghost-like regions with faint staining',
                   'Dark clusters of tumor cells: concentrated, intensely stained regions indicating dense metastases',
                   'Altered stromal staining: surrounding stroma may appear more pink or irregularly colored due to reactive changes or fibrosis']
        tumor_4 = ['Irregular clustering of tumor cells disrupting normal tissue organization', 
                   'High cellular density regions compared to surrounding normal lymphoid tissue',
                   'Loss of normal follicular or sinusoidal structures in affected areas',
                   'Heterogeneous texture patterns with abrupt transitions between normal and abnormal regions',
                   'Streaks or nests of tumor cells forming elongated or irregular shapes',
                   'Patchy or fragmented tumor regions scattered within otherwise normal tissue',
                   'Diffuse infiltration patterns, where tumor cells spread along existing tissue frameworks',
                   'Sharp contrast in color intensity and texture between tumor and adjacent stroma',
                   'Micro-metastatic foci appearing as small, isolated clusters with distinct textures',
                   'Distorted vascular or connective tissue patterns due to tumor invasion']
        return [normal_1, normal_2, normal_3, normal_4], [tumor_1, tumor_2, tumor_3, tumor_4]
    
    elif dataset == 'tcga_nsclc':
        luad_1 = ['Glandular structure formation: Tumor cells are arranged in gland-like or small acinar structures', 
                  'Columnar or cuboidal cells: Tumor cells often appear columnar or cuboidal, resembling glandular epithelium',
                  'Enlarged and pleomorphic nuclei: Nuclei are often enlarged and irregular in shape, showing pleomorphism',
                  'Coarse, granular chromatin: Chromatin is dense and granular, with prominent nucleoli',
                  'Indistinct cell boundaries: Cell membranes are poorly defined, with reduced intercellular spaces',
                  'Vacuolated cytoplasm: Some cells show cytoplasmic vacuolation or mucinous changes',
                  'Intraluminal secretions: Glandular lumens often contain mucin or secretory material',
                  'Increased mitotic figures: Actively proliferating cells display noticeable mitoses',
                  'Cellular pleomorphism: Significant variation in cell size and shape is observed within the same region',
                  'Microvascular infiltration: Tumor cells may infiltrate along microvessels or form focal clusters']
        luad_2 = ['Glandular formation: Tumor cells often form irregular or well-formed glands', 
                  'Acinar structures: Small, rounded gland-like clusters of tumor cells',
                  'Papillary patterns: Finger-like projections lined by neoplastic cells',
                  'Lepidic growth: Tumor cells spreading along alveolar walls without stromal invasion',
                  'Mucin production: Presence of intracellular or extracellular mucin pools',
                  'Nuclear atypia: Enlarged, pleomorphic, hyperchromatic nuclei',
                  'Cytoplasmic vacuolation: Clear or foamy cytoplasm in some tumor cells',
                  'Fibrovascular cores: Supporting stroma within papillary structures',
                  'Cribriform pattern: Glandular structures with multiple lumina resembling Swiss cheese',
                  'Desmoplastic stroma: Dense fibrous tissue surrounding tumor nests']
        luad_3 = ['Glandular formation: well-defined glandular structures with luminal spaces', 
                  'Loose stroma: relatively less dense fibrous stroma compared to LUSC',
                  'Intra-tumoral lymphocyte infiltration: presence of scattered lymphocytes within tumor nests',
                  'Peritumoral lymphoid aggregates: clusters of immune cells surrounding tumor areas',
                  'Tumor-associated macrophages (TAMs): macrophages within stromal regions',
                  'Necrotic foci: small areas of cell death within tumor tissue',
                  'Angiogenesis: newly formed, thin-walled blood vessels in the tumor stroma',
                  'Mucin production: extracellular mucin pools or mucin-filled cells',
                  'Fibroblast activation: spindle-shaped stromal fibroblasts surrounding tumor glands',
                  'Heterogeneous immune cell distribution: variable density and types of immune cells across different tumor regions']
        luad_4 = ['Variable cell density: some regions are densely packed with tumor cells, while others are more sparsely populated', 
                  'Glandular formation heterogeneity: areas with well-formed glands coexist with poorly differentiated or irregular glandular structures',
                  'Nuclear size variation: nuclei differ in size and shape across different tumor regions',
                  'Necrotic regions: focal necrosis may appear in some areas but not others',
                  'Stromal distribution differences: some regions have abundant fibrotic stroma, others are more cellular',
                  'Immune cell infiltration variability: immune cells are unevenly distributed, with some hotspots and some “cold” areas',
                  'Mucin content heterogeneity: certain regions produce mucin, while others show little or none',
                  'Vascular density variation: blood vessel density differs across tumor regions',
                  'Tumor margin irregularity: invasive fronts show irregular, heterogeneous patterns compared to central tumor regions',
                  'Mitotic activity variation: areas with high mitotic figures coexist with regions showing lower proliferation']
        
        lusc_1 = ['Keratinization: presence of keratin pearls or individual cell keratinization',
                  'Intercellular bridges: visible desmosomal connections between tumor cells',
                  'Polygonal tumor cells: cells often have a squarish or polygonal shape',
                  'Dense cytoplasm: cells show abundant eosinophilic cytoplasm',
                  'Nuclear pleomorphism: variability in nuclear size and shape',
                  'Hyperchromatic nuclei: nuclei are darkly stained due to increased chromatin',
                  'Prominent nucleoli: large, noticeable nucleoli within tumor nuclei',
                  'High mitotic activity: frequent mitotic figures indicating rapid proliferation',
                  'Individual cell keratinization: keratin within single tumor cells, not just in pearls',
                  'Scant glandular differentiation: minimal or absent gland formation, distinguishing LUSC from LUAD']
        lusc_2 = ['Keratin pearls: concentric layers of keratinized cells forming characteristic whorls',
                  'Intercellular bridges: visible desmosomal connections between squamous cells',
                  'Polygonal tumor cells: cells with well-defined, often angular shapes',
                  'Squamous differentiation: areas showing maturation toward squamous epithelium',
                  'Nuclear pleomorphism: variability in nuclear size and shape within tumor regions',
                  'High mitotic activity: frequent mitotic figures indicating rapid proliferation',
                  'Central necrosis: localized necrotic areas within tumor nests',
                  'Dense cellularity: tightly packed tumor cells with minimal stroma',
                  'Keratinization of individual cells: single cells showing eosinophilic keratin',
                  'Irregular tumor nests: sheets or nests of tumor cells with jagged borders infiltrating surrounding tissue']
        lusc_3 = ['Dense tumor nests: tightly packed clusters of malignant squamous cells',
                  'Keratin pearls: concentric layers of keratinized cells within tumor nests',
                  'Intercellular bridges: visible connections between squamous tumor cells',
                  'Fibrotic stroma: abundant fibrous tissue surrounding tumor clusters',
                  'Inflammatory infiltrates: significant presence of lymphocytes and other immune cells',
                  'Necrotic regions: areas of cell death within tumor masses',
                  'Angiogenesis: increased formation of new blood vessels in the tumor microenvironment',
                  'Peritumoral edema: localized swelling in the stroma surrounding tumor nests',
                  'Tumor-stroma interface heterogeneity: irregular boundaries between tumor cells and stroma',
                  'Immune cell aggregates: formation of tertiary lymphoid structures near tumor areas']
        lusc_4 = ['Dense clusters of tumor cells interspersed with small stromal regions',
                  'Irregular nests of squamous cells** forming keratin pearls in localized areas',
                  'Regions of necrosis surrounded by viable tumor cells',
                  'Variable nuclear size and shape across different tumor regions',
                  'Patchy infiltration of immune cells, unevenly distributed within the tumor',
                  'Fibrotic stroma areas alternating with high cellular density regions',
                  'Heterogeneous keratinization, with some regions highly keratinized and others minimally so',
                  'Local variation in mitotic activity, showing hotspots of proliferating cells',
                  'Microcystic or small gland-like structures scattered in certain regions',
                  'Spatially variable tumor margins, with some areas showing well-defined borders and others infiltrative growth into surrounding tissue']
        
    
        return [luad_1, luad_2, luad_3, luad_4], [lusc_1, lusc_2, lusc_3, lusc_4]
    
    elif dataset == 'tcga_brca':
        idc_1 = ['Pleomorphic nuclei: nuclei vary in size and shape',
                 'Increased nuclear-to-cytoplasmic ratio: nuclei appear relatively large compared to cytoplasm',
                 'Prominent nucleoli: nucleoli are easily visible and often enlarged',
                 'Hyperchromatic nuclei: nuclei are deeply stained due to dense chromatin',
                 'Mitotic figures: frequent cell division visible under the microscope',
                 'Irregular nuclear membranes: nuclear borders are uneven or jagged',
                 'Anisocytosis: variation in cell size among tumor cells',
                 'Anisonucleosis: variation in nuclear size among tumor cells',
                 'High cellular density: cells are closely packed with little intervening stroma',
                 'Loss of polarity: cells lose the normal orientation relative to the duct structure']
        idc_2 = ['Formation of irregular duct-like structures** within the tumor',
                 'Presence of solid tumor nests** without clear lumen',
                 'Tubular or acinar patterns resembling normal ducts but distorted',
                 'Central necrosis in some tumor glands',
                 'Cribriform architecture, where tumor cells form sieve-like spaces',
                 'Prominent desmoplastic stroma surrounding tumor clusters',
                 'Irregular gland contours with jagged or serrated edges',
                 'Lumen formation with cellular debris inside ducts',
                 'Haphazard arrangement of tumor glands, lacking organized polarity',
                 'Fused or coalescing: duct structures, forming large irregular masses']
        idc_3 = ['Desmoplastic reaction: dense fibrous tissue surrounding tumor nests',
                 'Collagen deposition: increased collagen fibers in the stroma',
                 'Fibroblast proliferation: activated fibroblasts forming tumor-associated stroma',
                 'Myofibroblast infiltration: spindle-shaped myofibroblasts contributing to ECM remodeling',
                 'Peritumoral fibrosis: thickened fibrotic bands around invasive tumor areas',
                 'Inflammatory cell infiltration: presence of lymphocytes, macrophages, or plasma cells in the stroma',
                 'Edematous stroma: areas of loose, watery stroma with increased interstitial space',
                 'Angiogenesis in stroma: newly formed small blood vessels supporting tumor growth',
                 'Hyalinization: localized glassy, eosinophilic ECM deposits',
                 'Tumor-stroma interface heterogeneity: irregular boundaries between tumor clusters and surrounding stroma, reflecting invasive behavior']
        idc_4 = ['Well-defined tumor nests: IDC often forms relatively cohesive clusters of tumor cells',
                 'Irregular tumor borders: The margins may be jagged due to variable invasion into surrounding tissue',
                 'Focal infiltration: Tumor cells infiltrate locally in discrete patches rather than diffusely',
                 'Tubular or duct-like structures: Some areas retain glandular or ductal architecture within the tumor mass',
                 'Desmoplastic stroma: Dense fibrous tissue often surrounds the tumor clusters',
                 'Micropapillary projections: Small papillary structures may extend into surrounding tissue',
                 'Periductal invasion: Tumor cells can spread along pre-existing ducts',
                 'Heterogeneous density: Tumor cell density varies within different regions of the same lesion',
                 'Focal necrosis: Small areas of cell death can appear inside the tumor mass',
                 'Local lymphovascular invasion: Tumor cells occasionally invade small vessels near the primary mass']
        ilc_1 = ['Small, uniform cells: cells are generally smaller and more consistent in size',
                 'Elongated or oval nuclei: nuclei often appear narrow and oval-shaped',
                 'Minimal nuclear pleomorphism: low variation in nuclear size and shape',
                 'Indented or “coffee-bean” nuclei: occasional nuclei show longitudinal grooves',
                 'Sparse cytoplasm: cells have relatively scant cytoplasm',
                 'Single-file arrangement: cells often infiltrate in linear chains between collagen fibers',
                 'Loose cell cohesion: weak intercellular connections due to loss of E-cadherin',
                 'Low mitotic activity: fewer visible mitotic figures compared to IDC',
                 'Round or slightly elongated nucleoli: nucleoli are small and not prominent',
                 'Discohesive infiltrative pattern: cells spread diffusely without forming glandular structures']
        ilc_2 = ['Single-file cell arrangement: tumor cells infiltrate in linear “Indian file” patterns',
                 'Loss of cohesive glandular structures: glands are poorly formed or absent',
                 'Targetoid pattern around normal ducts: tumor cells surround normal ductal structures in concentric rings',
                 'Small, uniform tumor cells: nuclei are often round and relatively bland',
                 'Minimal desmoplastic response: stroma is often loose rather than fibrotic',
                 'Diffuse infiltration: tumor cells spread in a more scattered, non-nodular manner',
                 'Signet ring cells: some tumor cells show cytoplasmic mucin displacing the nucleus',
                 'Trabecular growth: thin cords of cells infiltrate between normal tissue elements',
                 'Low mitotic activity in some regions: compared to IDC, cell division is often less prominent',
                 'Loss of E-cadherin expression: leading to reduced cell-to-cell adhesion and the characteristic dispersed pattern']
        ilc_3 = ['Loose, fibrous stroma surrounding the tumor cells rather than dense desmoplastic reaction',
                 'Minimal inflammatory infiltration, with fewer lymphocytes compared to IDC',
                 'Collagen fibers arranged in linear or parallel patterns along infiltrating tumor cells',
                 'Abundant extracellular mucin or mucin-like material in some areas',
                 'Reduced stromal cellularity, giving a “cleaner” or more acellular background',
                 'Stroma shows subtle edema or myxoid changes, often creating a slightly pale appearance',
                 'Tumor cells infiltrate stroma individually or in single-file rows, leaving ECM largely intact',
                 'Minimal fibroblast activation, with fewer myofibroblasts than in IDC-associated stroma',
                 'Sparse vascular proliferation, with fewer new capillaries formed in ECM',
                 'Stromal collagen often shows wavy or serpentine patterns, reflecting the diffuse, infiltrative nature of ILC']
        ilc_4 = ['Single-file infiltration: tumor cells often invade in linear chains rather than clusters',
                 'Targetoid pattern: around ducts tumor cells may wrap around normal ductal structures',
                 'Diffuse infiltration: tumor spreads widely without forming a solid mass',
                 'Minimal desmoplastic reaction: the surrounding stroma shows little fibrotic response',
                 'Indistinct tumor margins: boundaries between tumor and normal tissue are poorly defined',
                 'Scattered tumor cell clusters: small groups of cells are dispersed within stroma.',
                 'Perivascular infiltration: tumor cells infiltrate along small vessels',
                 'Subtle stromal expansion: tissue architecture is mildly disrupted, often overlooked',
                 'Occasional “Indian file” cords: narrow linear arrangements resembling beads on a string',
                 'Heterogeneous density: some regions have sparse tumor cells, others slightly denser, contributing to irregular spatial patterns']
        
        return [idc_1, idc_2, idc_3, idc_4], [ilc_1, ilc_2, ilc_3, ilc_4]




if __name__ == "__main__":
    # 加载模型
    llm_model = 'deepseek'
    
    if llm_model == 'qwen':
        model_path = "/data3/LLM_weights/Qwen_2_5_7B_Instruct_Int4"
    elif llm_model == 'llama':
        model_path = '/data3/LLMs/LLM-Research/Llama-3.2-1B-Instruct'
    elif llm_model == 'deepseek':
        model_path = '/data3/LLMs/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
        
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float32,device_map="auto")
    model.eval()
    
    dataset = 'tcga_brca'
    
    if dataset == 'camelyon_all':
        class_name = ['normal lymph node', 'metastatic lymph node']
    # guide = """For the diagnosis of breast cancer lymph node metastasis, what visually descriptive features characterize {class name} at both low and high resolutions within the whole-slide image? Please summarize into a single paragraph."""
        guide = """What are the visually descriptive characteristics of {class name} in the whole slide image? Please summarize into a single paragraph."""
        store_path = 'text_prompt/camelyon_all/'
        normal_descripts, tumor_descripts = CoT_and_ICL(dataset)
        
    
    elif dataset == 'tcga_nsclc':
        class_name = ['lung adenocarcinoma', 'lung squamous']
    # guide = """For the diagnosis of breast cancer lymph node metastasis, what visually descriptive features characterize {class name} at both low and high resolutions within the whole-slide image? Please summarize into a single paragraph."""
        guide = """What are the visually descriptive characteristics of {class name} in the whole slide image? Please summarize into a single paragraph."""
        store_path = 'text_prompt/tcga_nsclc/'
        normal_descripts, tumor_descripts = CoT_and_ICL(dataset)
    
    elif dataset == 'tcga_brca':
        class_name = ['invasive ductal carcinoma', 'invasive lobular carcinoma']
        store_path = 'text_prompt/tcga_brca/'
        normal_descripts, tumor_descripts = CoT_and_ICL(dataset)
        
    elif dataset == 'kidney':
        class_name = ['focal proliferative IgA nephropathy', 'focal sclerosing IgA nephropathy', 'mild mesangial proliferative IgA nephropathy', 
                      'proliferative-sclerosing IgA nephropathy', 'focal proliferative-necrotizing IgA nephropathy', 'minimal change type IgA nephropathy']
        store_path = 'text_prompt/kidney/'
    
    # guide = """What are the visually descriptive characteristics of {class name} in the whole slide image? Please summarize into a single paragraph."""
    
    # for idx, cls_name in enumerate(class_name):
    #     user_prompt = guide.replace("{class name}", cls_name)
    #     #print(user_prompt)
    #     responses = []
    #     for i in range(100):
    #         response = qwen(user_prompt, tokenizer, model, "Now you are pathologist")
    #         responses.append(response)
    #     df = pd.DataFrame(responses)
    #     df.to_csv(store_path + 'generated_{}.csv'.format(int(idx)))
    
    guide = 'To supplement and systematize the description of {class name} in high-resolution imaging from four perspectives: {perspectives}. Please summarize into a single paragraph.'
    for idx, cls_name in enumerate(class_name):
        user_prompt = guide.replace("{class name}", cls_name)
        responses = []
        for i in range(300):
            if idx == 0:
                st = ''
                for idxi, i in enumerate(normal_descripts):
                    if idxi == 0:
                        st = random.sample(i, 1)[0]
                    else:
                        st = st + ', ' + random.sample(i, 1)[0]
            elif idx == 1:
                st = ''
                for idxi, i in enumerate(tumor_descripts):
                    if idxi == 0:
                        st = random.sample(i, 1)[0]
                    else:
                        st = st + ', ' + random.sample(i, 1)[0]
            prompt = user_prompt.replace("{perspectives}", st)
            response = qwen(prompt, tokenizer, model, "Now you are pathologist")
            responses.append(response)
        df = pd.DataFrame(responses)
        
        if llm_model == 'qwen':
            df.to_csv(store_path + 'generated_new_{}.csv'.format(int(idx)))
        elif llm_model == 'llama':
            df.to_csv(store_path + 'generated_new_{}_llama.csv'.format(int(idx)))
        elif llm_model == 'deepseek':
            df.to_csv(store_path + 'generated_new_{}_deepseek.csv'.format(int(idx)))
        
                
                
        
    
    
    
        
        
        
    
    
    
    

