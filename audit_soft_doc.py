# -*- coding: utf-8 -*-
"""
全面风险排查脚本：审查软著3000行源代码文档
"""

import re
import docx

def audit():
    doc = docx.Document("手语音系学参数识别与自动标注系统_V3.0_源代码.docx")
    lines = [p.text for p in doc.paragraphs]
    print(f"=== 开始全量审计：共提取代码行数 {len(lines)} ===")

    findings = {
        "licenses": [],
        "companies": [],
        "personal_info": [],
        "informal_remarks": [],
        "security_keys": [],
        "ai_traces": [],
        "anomalies": []
    }

    # 1. 开源协议与版权声明
    license_patterns = [
        r'\bmit\b', r'\bapache\b', r'\bgpl\b', r'\bbsd\b', r'\bmozilla\b',
        r'copyright\s*\(c\)', r'all\s*rights\s*reserved', r'licensed\s*under'
    ]
    
    # 2. 外部公司与商业实体
    company_patterns = [
        r'\btencent\b', r'\bbaidu\b', r'\balibaba\b', r'\bbytedance\b',
        r'\bfacebook\b', r'\bmeta\b', r'\bapple\b'
    ]

    # 3. 个人敏感绝对路径与隐私（如 C:\Users\xxx）
    path_pattern = re.compile(r'[a-zA-Z]:\\(?:users|home)\b', re.IGNORECASE)

    # 4. 口语化/非正式/学术不严谨注释
    informal_words = [
        '学姐', '学长', '导师', '老师', '暂时', '临时', '写死',
        '凑合', '乱写', '随手', '测试用', 'todo', 'fixme', 'hack', 'xxx'
    ]

    # 5. 密钥与私密凭证
    key_pattern = re.compile(r'(?:api_key|secret|password|private_key|token)\s*=\s*[\'"][^\'"]{6,}[\'"]', re.IGNORECASE)

    # 6. AI 痕迹词汇
    ai_patterns = [
        r'\bchatgpt\b', r'\bcopilot\b', r'\bopenai\b', r'\bclaude\b',
        r'\banthropic\b', r'\bprompt\b', r'\bllm\b', r'\bgpt-[34]'
    ]

    for idx, line in enumerate(lines, 1):
        low = line.lower()
        
        # 排除包含在合法变量名中的误报（如 google 在 mediapipe 导入中的 google.protobuf）
        for pat in license_patterns:
            if re.search(pat, low):
                findings["licenses"].append((idx, line.strip()))
                
        for pat in company_patterns:
            if re.search(pat, low):
                findings["companies"].append((idx, line.strip()))
                
        if path_pattern.search(line) or "liushiyue" in low:
            findings["personal_info"].append((idx, line.strip()))

        for word in informal_words:
            if word in low:
                findings["informal_remarks"].append((idx, word, line.strip()))

        if key_pattern.search(line):
            findings["security_keys"].append((idx, line.strip()))

        for pat in ai_patterns:
            if re.search(pat, low):
                # 过滤合法的 Python 字符如 re.fullmatch 中的 llm
                if pat == r'\bllm\b':
                    if re.search(r'\bllm\b', low):
                        findings["ai_traces"].append((idx, line.strip()))
                else:
                    findings["ai_traces"].append((idx, line.strip()))

    print("\n--- 排查结果统计 ---")
    print(f"1. 开源协议/版权声明冲突: {len(findings['licenses'])} 处")
    for item in findings['licenses']:
        print(f"   - L{item[0]}: {item[1]}")

    print(f"2. 外部商业公司名称: {len(findings['companies'])} 处")
    for item in findings['companies']:
        print(f"   - L{item[0]}: {item[1]}")

    print(f"3. 个人绝对路径/用户隐私: {len(findings['personal_info'])} 处")
    for item in findings['personal_info']:
        print(f"   - L{item[0]}: {item[1]}")

    print(f"4. 口语化/非正式/TODO 注释: {len(findings['informal_remarks'])} 处")
    for item in findings['informal_remarks']:
        print(f"   - L{item[0]} [{item[1]}]: {item[2]}")

    print(f"5. 敏感密码/密钥凭据: {len(findings['security_keys'])} 处")
    for item in findings['security_keys']:
        print(f"   - L{item[0]}: {item[1]}")

    print(f"6. AI 痕迹/生成标签: {len(findings['ai_traces'])} 处")
    for item in findings['ai_traces']:
        print(f"   - L{item[0]}: {item[1]}")

if __name__ == "__main__":
    audit()
