import os
import re
import random
import asyncio
import aiofiles
import aiohttp

async def remote_code_judge(response: str, label: str) -> float:
    """提取 C++ 代码 -> 提交到评测机 -> 等待结果 -> 返回 [0.0/1.0]"""
    # print("Hello from remote_code_judge")
    # print("Response received for judging:", response)
    # print("Problem label:", label)
    judgeHost = os.getenv("JUDGE_HOST", "LightCPVerifier")
    judgePort = os.getenv("JUDGE_PORT", "8081")
    judgeBaseUrl = f"http://{judgeHost}:{judgePort}"

    code, lang = extract_code(response)
    if not code:
        return 0.0

    sid = await _submit_code(pid = label, lang = lang, code = code, baseUrl = judgeBaseUrl)
    if not sid:
        return 0.0
    
    # try:
    #     # 定义一个日志目录
    #     log_dir = os.path.expanduser("~/judge_response_logs")
    #     # os.makedirs 是同步的，但通常只在启动时执行一次，这里为了方便
    #     # 如果追求极致性能，此步骤应在服务启动时完成
    #     os.makedirs(log_dir, exist_ok=True) 
        
    #     log_path = os.path.join(log_dir, f"{sid}_response.txt")
        
    #     # 使用 aiofiles 进行异步文件写入，避免阻塞
    #     async with aiofiles.open(log_path, "w", encoding="utf-8") as f:
    #         await f.write(f"--- Problem Label: {label} ---\n")
    #         await f.write(f"--- Submission ID: {sid} ---\n\n")
    #         await f.write(response)

    # except Exception as e:
    #     # 写入日志失败不应中断主流程，打印一个警告即可
    #     print(f"[Judge-Warning] Failed to write response log to {log_path}: {e}")
    #     # --- 增加的日志记录代码块 [结束] ---

    result = await _wait_for_result(sid, judgeBaseUrl)
    return _calc_score(result)

def extract_code(response: str) -> tuple[str, str]:
    """
    从LLM的API响应中提取最后的代码答案并识别语言
    
    Args:
        response: LLM的完整响应文本
        
    Returns:
        tuple: (代码内容, 语言类型) 
               语言类型为 "cpp" 或 "py"
               如果没找到代码返回 ("", "py")
    """
    
    # 查找所有代码块（按出现顺序）
    code_blocks = []

    # 匹配无语言标识的代码块
    pattern_no_lang = r"```\n(.*?)\n```"
    matches = re.finditer(pattern_no_lang, response, re.DOTALL)
    for match in matches:
        code = match.group(1).strip()
        # 通过代码内容推断语言
        lang = _detect_language(code)
        code_blocks.append((code, lang))
    
    # 匹配带语言标识的代码块
    pattern_with_lang = r"```(\w+)\n(.*?)\n```"
    matches = re.finditer(pattern_with_lang, response, re.DOTALL)
    for match in matches:
        lang = match.group(1).lower()
        code = match.group(2).strip()
        code_blocks.append((code, lang))
    
    # 如果没找到任何代码块
    if not code_blocks:
        return "", "py"
    
    # 返回最后一个代码块（最终答案）
    final_code, detected_lang = code_blocks[-1]
    
    # 标准化语言名称
    if detected_lang in ["cpp", "c++", "cxx"]:
        return final_code, "cpp"
    elif detected_lang in ["python", "py"]:
        return final_code, "py"
    else:
        # 如果语言不确定，通过代码内容再次判断
        inferred_lang = _detect_language(final_code)
        return final_code, inferred_lang

def _detect_language(code: str) -> str:
    code_lower = code.lower()
    cpp_indicators = [
        "#include", "using namespace std", "std::", 
        "cout<<", "cin>>", "endl", "vector<", "#define"
    ]   
    cpp_score = sum(1 for indicator in cpp_indicators if indicator in code_lower)
    if cpp_score > 0:
        return "cpp"
    else:
        return "py"


async def _submit_code(pid: str, code: str, baseUrl: str, lang: str = "cpp"):
    """提交代码"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"pid": pid, "lang": lang, "code": code}
            async with session.post(f"{baseUrl}/submit", json=payload) as r:
                if r.status == 200:
                    return (await r.json()).get("sid")
    except Exception as e:
        print(f"Submit failed: {e}")
    return None


def _calc_score(result) -> float:
    if not result:
        return 0.0
    return 1.0 if result.get("passed", False) else 0.0

async def _wait_for_result(sid: int, baseUrl: str, timeout: int = 30):
    """等待结果"""
    try:
        async with aiohttp.ClientSession() as session:
            for _ in range(timeout):
                async with session.get(f"{baseUrl}/result/{sid}") as r:
                    if r.status == 200:
                        data = await r.json()
                        if data.get("status") == "done":
                            return data
                await asyncio.sleep(1)
    except Exception as e:
        print(f"Query failed: {e}")
    return None

async def __test__():
    response = """Here is a sample C++ code:
```cpp
#include <iostream>
using namespace std;
int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
```"""
    code, lang = extract_code(response)
    print(f"Extracted Code:\n{code}\nLanguage: {lang}")
    res = await remote_code_judge(response, "2018B")
    print(res)

if __name__ == "__main__":
    asyncio.run(__test__())