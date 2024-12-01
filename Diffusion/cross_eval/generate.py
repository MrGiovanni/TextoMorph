import random

# 定义关键词列表
keywords = [
    "cyst", "hypoattenuating", "hypodensity", "hyperattenuating",
    "heterogeneous enhancement", "arterial enhancement", "washout",
    "cirrhosis", "metastases", "ill-defined", "well-defined"
]

# 定义句式模板
sentence_templates = [
    "{keyword} is detected.",
    "{keyword} is visible on imaging.",
    "Imaging reveals {keyword}.",
    "The scan shows {keyword}.",
    "The scan confirms {keyword}.",
    "{keyword} is reported.",
    "Imaging findings include {keyword}.",
    "{keyword} is observed.",
    "The imaging study shows {keyword}.",
    "There is evidence of {keyword}.",
    "The scan demonstrates {keyword}.",
    "Imaging highlights {keyword}.",
    "The report notes {keyword}.",
    "{keyword} is seen.",
    "The scan reveals {keyword}.",
    "The report mentions {keyword}.",
    "{keyword} is found on the scan.",
    "Imaging provides evidence of {keyword}.",
    "The radiology report confirms {keyword}.",
    "{keyword} is present on the scan.",
    "{keyword} is noted on the report.",
    "The scan verifies {keyword}.",
    "Imaging supports findings of {keyword}.",
    "{keyword} is conclusively observed.",
    "The scan illustrates {keyword}.",
    "There are indications of {keyword}.",
    "The scan highlights features of {keyword}.",
    "The report discusses {keyword}.",
    "A detailed scan shows {keyword}.",
    "There is confirmed presence of {keyword}.",
    "{keyword} is a prominent finding.",
    "{keyword} is unmistakably present.",
    "{keyword} is highlighted in the imaging study.",
    "The scan provides clear evidence of {keyword}.",
    "There is imaging evidence for {keyword}.",
    "Findings strongly suggest {keyword}.",
    "The imaging demonstrates {keyword}.",
    "Findings are consistent with {keyword}.",
    "{keyword} is confirmed via scan.",
    "The radiologist highlights {keyword}.",
    "{keyword} is noted in this study.",
    "The diagnosis includes {keyword}.",
    "{keyword} findings are supported by imaging.",
    "The report describes {keyword}.",
    "Radiological evidence supports {keyword}.",
    "{keyword} is a primary feature of this scan.",
    "{keyword} is a secondary feature of the findings.",
    "The overall findings reveal {keyword}.",
    "The findings point to {keyword}.",
    "Imaging further confirms {keyword}.",
    "{keyword} is demonstrated clearly in this case.",
    "Clinical imaging findings align with {keyword}.",
    "There is prominent visibility of {keyword}.",
    "This imaging case shows {keyword}.",
    "No significant evidence contradicts {keyword}.",
    "Overall findings corroborate {keyword}.",
    "{keyword} is a noted observation.",
    "The observations confirm {keyword}.",
    "Clear indications of {keyword} are seen.",
    "Strong imaging results show {keyword}.",
    "{keyword} is evident.",
    "The image unmistakably shows {keyword}.",
    "Definitive findings include {keyword}.",
    "{keyword} appears prominently.",
    "{keyword} is indisputable in the imaging.",
    "Further imaging reveals {keyword}.",
    "Additional findings include {keyword}.",
    "The findings are supportive of {keyword}.",
    "Observed features correspond to {keyword}.",
    "The radiological description matches {keyword}.",
    "Visual evidence points to {keyword}.",
    "Overall imaging indicates {keyword}.",
    "The scan aligns with features of {keyword}.",
    "Clear visibility of {keyword} is noted.",
    "Radiology highlights characteristics of {keyword}.",
    "The imaging findings revolve around {keyword}.",
    "{keyword} is central to the findings.",
    "Detailed imaging elaborates on {keyword}.",
    "Critical features show {keyword}.",
    "Radiological data underscores {keyword}.",
    "Diagnostic imaging corroborates {keyword}.",
    "The scan solidifies findings of {keyword}.",
    "Observations distinctly show {keyword}.",
    "Radiological imaging confirms {keyword}.",
    "The radiologist emphasizes {keyword}.",
    "Enhanced imaging highlights {keyword}.",
    "This diagnostic study shows {keyword}.",
    "Comprehensive imaging finds {keyword}.",
    "The report establishes findings of {keyword}.",
]

# 从文件读取数据
def read_file(file_path):
    with open(file_path, "r") as f:
        return f.readlines()

# 提取关键词
def extract_keyword(sentences):
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence.lower():
                return keyword
    return None

# 扩展句子
def expand_sentences_for_line(line, num_new_sentences=90):
    parts = line.strip().split("          ")
    if len(parts) < 3:
        return line  # 格式不符，跳过

    id1, id2, *original_sentences = parts
    keyword = extract_keyword(original_sentences)
    if not keyword:
        print(f"Warning: No valid keyword found in line: {line}")
        return line

    generated_sentences = set(original_sentences)
    while len(generated_sentences) < len(original_sentences) + num_new_sentences:
        try:
            template = random.choice(sentence_templates)
            new_sentence = template.format(keyword=keyword)
            generated_sentences.add(new_sentence)
        except Exception as e:
            print(f"Error generating sentence: {e}")
            break

    expanded_line = f"{id1} {id2} " + "          ".join(list(generated_sentences))
    return expanded_line

# 保存文件
def save_file(file_path, lines):
    with open(file_path, "w") as f:
        f.writelines(lines)

# 检测句子数量
def check_sentence_count(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        parts = line.strip().split("          ")
        sentence_count = len(parts) - 2
        print(f"Line {i + 1}: {sentence_count} sentences")
        if sentence_count != 100:
            print(f"Warning: Line {i + 1} has {sentence_count} sentences (expected 100)")

# 主函数
def main():
    input_file = "/ccvl/net/ccvl15/xinran/TextoMorph/TextoMorph/Diffusion/cross_eval/kits_tumor_data_early_fold/real_tumor_train_0.txt"
    output_file = "/ccvl/net/ccvl15/xinran/TextoMorph/TextoMorph/Diffusion/cross_eval/kits_tumor_data_early_fold/real_tumor_train_0_ok.txt"

    lines = read_file(input_file)
    expanded_lines = [expand_sentences_for_line(line) for line in lines]
    save_file(output_file, expanded_lines)
    print(f"Expanded lines saved to {output_file}")
    check_sentence_count(output_file)

# 执行主程序
if __name__ == "__main__":
    main()
