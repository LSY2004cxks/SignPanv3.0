# -*- coding: utf-8 -*-
"""
生成中国版权保护中心（CPCC）标准的软件著作权源程序文档
软件全称：手语音系学参数识别与自动标注系统
软件简称：SignPan
版本号：V3.0

规范要求：
1. 总页数：严格60页（前30页连续代码 + 后30页连续代码）。
2. 每页行数：严格50行代码，总计3000行。
3. 页眉格式：
   - 左侧：软件全称及版本号（手语音系学参数识别与自动标注系统 V3.0）
   - 右侧：页码格式（第 X 页 共 60 页）
   - 页眉下划线：单条浅灰色实线分割线
4. 排版规范：
   - A4纸张，页边距各2.0cm。
   - 字体：等宽代码字体 Consolas（中文字符宋体），字号 8.5pt。
   - 段落设置：0段前、0段后，固定行距 11.5pt，确保每页50行整齐美观且绝对不溢出。
   - 分页方式：每页第50行附加分页符，绝无多余空段。
"""

import os
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def clean_file_lines(filepath):
    """读取文件并清理连续多余空行，替换制表符为4空格"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()
    
    cleaned = []
    last_empty = False
    for line in raw_lines:
        line_str = line.rstrip('\r\n').replace('\t', '    ')
        if not line_str.strip():
            if not last_empty:
                cleaned.append('')
                last_empty = True
        else:
            cleaned.append(line_str)
            last_empty = False
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return cleaned

def add_field(p, field_text):
    """在段落中添加Word动态域（如 PAGE, NUMPAGES）"""
    run = p.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = field_text
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def generate_source_code_doc():
    # 1. 提取前1500行（系统调度 + 视频分割 + 特征提取核心算法）
    run_all_lines = clean_file_lines('run_all.py')
    segment_lines = clean_file_lines('segment_video.py')
    v2j_lines = clean_file_lines('video2json.py')
    
    first_pool = run_all_lines + segment_lines + v2j_lines
    if len(first_pool) < 1500:
        raise ValueError(f"前部分代码不足1500行，当前仅有 {len(first_pool)} 行")
    first_1500 = first_pool[:1500]

    # 2. 提取后1500行（Web标注交互服务 + 修正系统 + 结果导出引擎）
    gui_lines = clean_file_lines('gui_server.py')
    idx_load = gui_lines.index('def _load_corrections():')
    
    # 从 def _load_corrections(): 到结尾共 1499 行，补1行规范注释使之正好 1500 行
    last_1500 = [
        '# ==================== 标注管理、交互修正与多格式导出引擎 ===================='
    ] + gui_lines[idx_load:]
    
    if len(last_1500) != 1500:
        raise ValueError(f"后部分代码行数不等于1500，当前为 {len(last_1500)}")

    total_lines = first_1500 + last_1500
    assert len(total_lines) == 3000, f"总代码行数不等于3000，当前为 {len(total_lines)}"

    print("代码准备完毕：")
    print(f"  - 前30页代码（第1~1500行）：涵盖 run_all.py、segment_video.py、video2json.py")
    print(f"  - 后30页代码（第1501~3000行）：涵盖 gui_server.py 完整后端服务")
    print(f"  - 第1行：{first_1500[0]}")
    print(f"  - 第1500行：{first_1500[-1]}")
    print(f"  - 第1501行：{last_1500[0]}")
    print(f"  - 第3000行：{last_1500[-1]}")

    # 3. 创建 Word 文档
    doc = docx.Document()
    
    # 页面设置：A4纸，边距适度（上下2.0cm，左右2.0cm）
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.header_distance = Cm(1.2)
    section.footer_distance = Cm(1.2)

    # 4. 设置页眉
    header = section.header
    p_hdr = header.paragraphs[0]
    p_hdr.text = ""  # 清空默认空段
    
    # 页眉表格（1行2列），左边显示软件名及版本号，右边显示页码
    tbl = header.add_table(1, 2, Cm(17.0))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_l = tbl.cell(0, 0)
    cell_r = tbl.cell(0, 1)
    cell_l.width = Cm(11.5)
    cell_r.width = Cm(5.5)

    p_l = cell_l.paragraphs[0]
    p_l.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r_l = p_l.add_run("手语音系学参数识别与自动标注系统 V3.0")
    r_l.font.name = "宋体"
    r_l._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r_l.font.size = Pt(9)
    r_l.font.color.rgb = RGBColor(60, 60, 60)

    p_r = cell_r.paragraphs[0]
    p_r.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    r_r1 = p_r.add_run("第 ")
    r_r1.font.name = "宋体"
    r_r1._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r_r1.font.size = Pt(9)
    r_r1.font.color.rgb = RGBColor(60, 60, 60)
    
    add_field(p_r, "PAGE")
    
    r_r2 = p_r.add_run(" 页 共 ")
    r_r2.font.name = "宋体"
    r_r2._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r_r2.font.size = Pt(9)
    r_r2.font.color.rgb = RGBColor(60, 60, 60)
    
    add_field(p_r, "NUMPAGES")
    
    r_r3 = p_r.add_run(" 页")
    r_r3.font.name = "宋体"
    r_r3._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    r_r3.font.size = Pt(9)
    r_r3.font.color.rgb = RGBColor(60, 60, 60)

    # 移除表格所有内外边框，并添加下底线（软著页眉规范横线）
    tblPr = tbl._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="999999"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="none"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

    # 5. 写入代码：严格每页50行，共60页，每页第50行追加分页符
    font_name = "Consolas"
    font_size = Pt(8.5)
    line_spacing = Pt(11.5)

    lines_per_page = 50
    total_pages = 60

    for page_idx in range(total_pages):
        start_line = page_idx * lines_per_page
        end_line = start_line + lines_per_page
        page_lines = total_lines[start_line:end_line]

        for i, code_line in enumerate(page_lines):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = line_spacing

            # 避免空行被Word忽略高度
            display_text = code_line if code_line else " "
            run = p.add_run(display_text)
            run.font.name = font_name
            run._element.rPr.rFonts.set(qn('w:ascii'), font_name)
            run._element.rPr.rFonts.set(qn('w:hAnsi'), font_name)
            run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
            run.font.size = font_size
            run.font.color.rgb = RGBColor(30, 30, 30)

            # 该页最后一行（第50行）且不是最后一页，添加分页符
            if i == lines_per_page - 1 and page_idx < total_pages - 1:
                run.add_break(WD_BREAK.PAGE)

    output_filename = "手语音系学参数识别与自动标注系统_V3.0_源代码.docx"
    doc.save(output_filename)
    print(f"\n生成成功！已保存为：{output_filename}")
    print(f"文档总段落数：{len(doc.paragraphs)}（严格等于3000行代码）")

if __name__ == "__main__":
    generate_source_code_doc()
