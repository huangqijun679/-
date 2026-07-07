# skills.py
import requests
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import urllib.parse
# ==========================================
# 1. 定义技能注册表和装饰器
# ==========================================
SKILLS_REGISTRY = {}

def register_skill(t):
    """将带有 @tool 的函数注册到全局字典中"""
    SKILLS_REGISTRY[t.name] = t
    return t

# ==========================================
# 2. 具体的技能实现
# ==========================================
@register_skill
@tool
def bing_web_search(query: str) -> str:
    """
    使用 Bing 搜索引擎执行联网搜索，返回前 10 条相关结果。
    当用户需要查询最新信息、新闻、百科知识或任何模型自身知识库不包含的实时信息时，使用此工具。

    Args:
        query: 要搜索的关键词或问题。
    """

    # 伪装成真实的浏览器请求
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
    }

    # 构造 Bing 搜索 URL (URL 编码处理中文等特殊字符)
    url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Bing 的主流搜索结果通常被包裹在 class 为 'b_algo' 的 <li> 标签中
        results = soup.find_all('li', class_='b_algo')

        parsed_results = []
        for item in results:
            # 提取标题和链接
            h2 = item.find('h2')
            if not h2:
                continue
            a_tag = h2.find('a')
            if not a_tag:
                continue

            title = h2.text.strip()
            link = a_tag.get('href', '')

            # 提取摘要：Bing 的摘要通常在 <p> 标签或特定的 <div> 中
            snippet_tag = item.find('p') or item.find('div', class_='b_caption')
            snippet = snippet_tag.text.strip() if snippet_tag else "无摘要"

            parsed_results.append(f"【标题】{title}\n【链接】{link}\n【摘要】{snippet}")

            # 达到 10 条结果即可退出
            if len(parsed_results) >= 10:
                break

        if not parsed_results:
            return f"未找到关于 '{query}' 的标准搜索结果。可能是触发了 Bing 的防爬虫机制，或该关键词无相关网页。"

        return f"找到关于 '{query}' 的 {len(parsed_results)} 条搜索结果:\n\n" + "\n---\n".join(parsed_results)

    except requests.exceptions.RequestException as e:
        return f"执行 Bing 搜索时发生网络错误: {str(e)}"
    except Exception as e:
        return f"解析搜索结果时发生异常: {str(e)}"

@register_skill
@tool
def calculate_mortgage(loan_amount: float, interest_rate: float, years: int) -> float:
    """
    计算房屋贷款每月的还款额。
    当用户提到'房贷计算'、'月供多少'、'帮我算算贷款'时，使用此工具。

    Args:
        loan_amount: 贷款总额（单位：万元）
        interest_rate: 年利率（例如 0.045 代表 4.5%）
        years: 贷款年限（例如 30）
    """
    monthly_rate = interest_rate / 12
    months = years * 12
    loan_amount_yuan = loan_amount * 10000
    monthly_payment = (
        loan_amount_yuan
        * monthly_rate
        * (1 + monthly_rate) ** months
        / ((1 + monthly_rate) ** months - 1)
    )
    return round(monthly_payment, 2)




@register_skill
@tool
def search_weather(region: str) -> str:
    """
    Python 技能：模拟浏览器请求获取今日某地区的天气。

    :param region: 城市或地区名称（支持中文拼音或英文，如 "Beijing", "Hong Kong", "Shanghai"）
    :return: 包含地区、天气状况和温度的字符串
    """
    # 1. 构造请求头，完美伪装成常用的 Chrome 浏览器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'  # 告诉服务器我们希望接收中文（如果服务器支持）
    }

    # 2. 构造目标 URL。format=3 表示返回带有 Emoji 的精简单行天气格式
    url = f"https://wttr.in/{region}?format=3"

    try:
        # 3. 发起 GET 请求
        response = requests.get(url, headers=headers, timeout=10)

        # 检查 HTTP 响应状态码，如果不是 200 则抛出异常
        response.raise_for_status()

        # 4. 提取并清理返回的数据
        weather_result = response.text.strip()

        if not weather_result:
            return f"未能找到 {region} 的天气信息，请检查地区名称拼写。"

        return weather_result

    except requests.exceptions.RequestException as e:
        # 错误处理：网络中断或超时
        return f"搜索 {region} 天气时发生网络错误: {str(e)}"


@register_skill
@tool
def get_metal_prices(metals: list[str] = None) -> str:
    """
    查询今日常见贵金属/基本金属的最新价格（基于国际期货市场）。
    当用户询问“今天金价多少”、“白银现在什么价格”、“帮我查查铜价”等问题时，使用此工具。

    Args:
        metals: 可选。要查询的金属名称列表（例如 ["黄金", "白银"]）。如果留空，将默认查询所有支持的常见金属。
    """
    import requests

    # 新浪财经国际期货接口标识与单位映射 (主要为 COMEX 数据)
    metal_symbols = {
        "黄金": ("hf_XAU", "美元/盎司"),
        "白银": ("hf_XAG", "美元/盎司"),
        "铜": ("hf_CAD", "美元/磅"),
        "铂金": ("hf_XPT", "美元/盎司"),
        "钯金": ("hf_XPD", "美元/盎司")
    }

    # 如果模型没有传入具体金属，则默认查询所有
    if not metals:
        metals = list(metal_symbols.keys())

    # 提取需要请求的代号
    query_symbols = []
    for metal in metals:
        if metal in metal_symbols:
            query_symbols.append(metal_symbols[metal][0])
        else:
            query_symbols.append(None)  # 用 None 占位表示不支持

    valid_symbols = [s for s in query_symbols if s]
    if not valid_symbols:
        return "未能识别任何支持的金属名称。目前支持: " + ", ".join(metal_symbols.keys())

    # 构造请求 URL，新浪财经支持逗号分隔批量请求
    url = "https://hq.sinajs.cn/list=" + ",".join(valid_symbols)

    # 【非常重要】必须提供 Referer，否则新浪会拦截防盗链，返回 403 Forbidden
    headers = {
        "Referer": "https://finance.sina.com.cn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 新浪接口返回的 JS 脚本数据使用 GBK 编码
        response.encoding = 'gbk'
        text = response.text

        results = []
        # 去除空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        line_idx = 0

        for metal, symbol in zip(metals, query_symbols):
            if not symbol:
                results.append(f"【{metal}】暂不支持该金属查询，目前支持: {', '.join(metal_symbols.keys())}")
                continue

            if line_idx < len(lines):
                line = lines[line_idx]
                line_idx += 1

                # 正常返回值格式示例：var hq_str_hf_XAU="1932.10,1931.20,...";
                if '="' in line:
                    # 提取双引号内的数据
                    data_str = line.split('="')[1].strip('";')
                    parts = data_str.split(',')

                    # 国际期货数据用逗号分隔，第0位是最新价，第3位最高，第4位最低
                    if len(parts) >= 5:
                        latest_price = parts[0]
                        high_price = parts[3]
                        low_price = parts[4]
                        unit = metal_symbols[metal][1]

                        results.append(
                            f"【{metal}】最新价: {latest_price} {unit} (今日最高: {high_price}, 最低: {low_price})")
                    else:
                        results.append(f"【{metal}】获取数据格式异常。")
                else:
                    results.append(f"【{metal}】暂无有效数据。")

        return "\n".join(results)

    except requests.exceptions.RequestException as e:
        return f"查询金属价格时发生网络请求错误: {str(e)}"

@register_skill
@tool
def remote_linux_executor_by_password(
    hostname: str, username: str, password: str, command: str, port: int = 22
) -> str:
    """
    通过 SSH（用户名和密码）远程连接 Linux 服务器，执行指定的 Shell 命令，并返回执行结果。

    Args:
        hostname: 服务器的 IP 地址或域名 (例如: "192.168.1.100")
        username: 登录用户名 (例如: "root")
        password: 登录密码
        command: 要在远程服务器上执行的 Shell 命令 (例如: "df -h")
        port: SSH 端口号，默认是 22
    """
    try:
        import paramiko

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=hostname,
            port=port,
            username=username,
            password=password,
            timeout=10,
            look_for_keys=False,
            allow_agent=False,
        )
        try:
            stdin, stdout, stderr = ssh.exec_command(command, timeout=45, get_pty=False)
            out_res = stdout.read().decode("utf-8", errors="ignore")
            err_res = stderr.read().decode("utf-8", errors="ignore")
            parts = []
            if out_res.strip():
                parts.append("【标准输出】\n" + out_res)
            if err_res.strip():
                parts.append("【错误输出】\n" + err_res)
            return "\n".join(parts) if parts else "命令执行完成，无输出"
        finally:
            ssh.close()
    except Exception as e:
        return f"【SSH连接或执行异常】: {str(e)}"


@register_skill
@tool
def remote_linux_executor_by_key(
    hostname: str, username: str, key_path: str, command: str, port: int = 22
) -> str:
    """
    通过 SSH 密钥对远程连接 Linux 服务器，执行指定的 Shell 命令，并返回执行结果。

    Args:
        hostname: 服务器的 IP 地址或域名 (例如: "192.168.1.100")
        username: 登录用户名 (例如: "root")
        key_path: 本地私钥文件路径 (例如: "~/.ssh/id_rsa")
        command: 要在远程服务器上执行的 Shell 命令 (例如: "df -h")
        port: SSH 端口号，默认是 22
    """
    try:
        import paramiko
        from pathlib import Path

        key_file = Path(key_path).expanduser()
        if not key_file.exists():
            return f"【错误】私钥文件不存在: {key_file}"

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=hostname,
            port=port,
            username=username,
            key_filename=str(key_file),
            timeout=10,
            look_for_keys=False,
            allow_agent=True,
        )
        try:
            stdin, stdout, stderr = ssh.exec_command(command, timeout=45, get_pty=False)
            out_res = stdout.read().decode("utf-8", errors="ignore")
            err_res = stderr.read().decode("utf-8", errors="ignore")
            parts = []
            if out_res.strip():
                parts.append("【标准输出】\n" + out_res)
            if err_res.strip():
                parts.append("【错误输出】\n" + err_res)
            return "\n".join(parts) if parts else "命令执行完成，无输出"
        finally:
            ssh.close()
    except Exception as e:
        return f"【SSH连接或执行异常】: {str(e)}"