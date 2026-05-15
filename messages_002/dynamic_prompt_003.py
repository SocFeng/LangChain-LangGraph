from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from common.model import llm_openai


# 携带的上下文数据context
class Context(TypedDict):
    user_role: str


@dynamic_prompt
def dynamicChangePromptByContent(request: ModelRequest) -> str:
    base_prompt = """你是一个数学学习助手"""
    context = request.runtime.context or {}
    role = context.get("user_role", "user")
    if role == "expert":
        return f"""{base_prompt}，提供详细的技术回应！解释后面要加上'====专家===='"""
    elif role == "beginner":
        return f"""{base_prompt}，请提供最简单容易听的懂的解释！解释后面要加上'====初学者===='"""
    else:
        return f"""{base_prompt}，解释后面要加上 ‘=====标准====’‘"""


# 创建agent执行器
agent = create_agent(
    model=llm_openai,
    tools=[],
    middleware=[dynamicChangePromptByContent],
    context_schema=Context,
)

# 使用
print("======================================初学者=================")
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "解释一下什么是韦达定理！"}]},
    context={"user_role": "beginner"}
)
print(resp)
"""
'韦达定理其实就是一句话：**“不用解方程，也能直接知道方程两个根的和与积。”**
拿最常见的**一元二次方程** `ax² + bx + c = 0`（a≠0）举例，假设它的两个解（根）是 `x₁` 和 `x₂`，韦达定理直接给出：
🔹 **两根相加**：`x₁ + x₂ = -b/a`  （⚠️注意有个负号！）
🔹 **两根相乘**：`x₁ × x₂ = c/a`

💡 **为什么它简单又好用？**
比如题目说“方程的两根之和是5，两根之积是6”，你根本不需要去算求根公式，脑子里马上就能反应出这两个数是2和3。它就像方程的“快捷查询键”，把方程表面的系数（a、b、c）和藏起来的答案（根）直接挂钩了。
📝 **初学者记忆口诀**：**“和反号，积原样”**（和是负的b除以a，积是c直接除以a）。
====初学者====
"""
print("======================================专家=================")
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "解释一下什么是韦达定理！"}]},
    context={"user_role": "export"}
)
print(resp)
"""
韦达定理（Vieta's Formulas）是代数学中揭示**多项式方程的根与系数之间内在联系**的重要定理，由16世纪法国数学家弗朗索瓦·韦达（François Viète）系统提出。
### 🔹 最常见形式（一元二次方程）
对于一元二次方程 \\( ax^2 + bx + c = 0 \\)（\\( a \\neq 0 \\)），设它的两个根为 \\( x_1, x_2 \\)（实根或复根均可），则：
- **两根之和**：\\( x_1 + x_2 = -\\dfrac{b}{a} \\)
- **两根之积**：\\( x_1 \\cdot x_2 = \\dfrac{c}{a} \\)

### 🔹 推广到 n 次方程
对于一般一元 n 次方程：
\\[ a_n x^n + a_{n-1} x^{n-1} + \\cdots + a_1 x + a_0 = 0 \\quad (a_n \\neq 0) \\]
设其 n 个根为 \\( x_1, x_2, \\dots, x_n \\)，则根与系数满足初等对称多项式关系：
- 一次对称和：\\( \\sum x_i = -\\dfrac{a_{n-1}}{a_n} \\)
- 两两乘积和：\\( \\sum_{i<j} x_i x_j = \\dfrac{a_{n-2}}{a_n} \\)
- 三三乘积和：\\( \\sum_{i<j<k} x_i x_j x_k = -\\dfrac{a_{n-3}}{a_n} \\)
- ⋯⋯
- 全部根的乘积：\\( x_1 x_2 \\cdots x_n = (-1)^n \\dfrac{a_0}{a_n} \\)
（规律：符号正负交替，系数比按降幂顺序对应各阶对称和）

### 🔹 为什么重要？
1. **免解方程**：无需求根公式，直接写出根的和与积。
2. **验根与构造**：已知根可快速写出方程；已知部分根可结合定理求未知根。
3. **对称式求值**：处理 \\( x_1^2+x_2^2 \\)、\\( \\frac{1}{x_1}+\\frac{1}{x_2} \\) 等表达式时，通过恒等变形转化为和与积。
4. **跨领域应用**：在函数图像、解析几何、不等式、数学竞赛中频繁出现。

### 🔹 小例子
方程 \\( 2x^2 - 5x + 3 = 0 \\)，由韦达定理：
- \\( x_1 + x_2 = -\\frac{-5}{2} = \\frac{5}{2} \\)
- \\( x_1 x_2 = \\frac{3}{2} \\)
若题目要求 \\( x_1^2 + x_2^2 \\)，可直接计算：
\\( x_1^2 + x_2^2 = (x_1+x_2)^2 - 2x_1x_2 = \\left(\\frac{5}{2}\\right)^2 - 2\\cdot\\frac{3}{2} = \\frac{25}{4} - 3 = \\frac{13}{4} \\)
💡 **注意**：韦达定理在复数域内恒成立，不依赖判别式 Δ 的正负；使用时务必保证最高次项系数 \\( a \\neq 0 \\)。
=====专家====
"""
print("======================================默认=================")

resp = agent.invoke(
    {"messages": [{"role": "user", "content": "解释一下什么是韦达定理！"}]},
)
print(resp)
"""
韦达定理（Vieta's Formulas）是代数学中揭示**多项式方程的根与系数之间内在联系**的重要定理，由16世纪法国数学家弗朗索瓦·韦达（François Viète）系统提出。
### 🔹 最常见形式（一元二次方程）
对于一元二次方程 \\( ax^2 + bx + c = 0 \\)（\\( a \\neq 0 \\)），设它的两个根为 \\( x_1, x_2 \\)（实根或复根均可），则：
- **两根之和**：\\( x_1 + x_2 = -\\dfrac{b}{a} \\)
- **两根之积**：\\( x_1 \\cdot x_2 = \\dfrac{c}{a} \\)

### 🔹 推广到 n 次方程
对于一般一元 n 次方程：
\\[ a_n x^n + a_{n-1} x^{n-1} + \\cdots + a_1 x + a_0 = 0 \\quad (a_n \\neq 0) \\]
设其 n 个根为 \\( x_1, x_2, \\dots, x_n \\)，则根与系数满足初等对称多项式关系：
- 一次对称和：\\( \\sum x_i = -\\dfrac{a_{n-1}}{a_n} \\)
- 两两乘积和：\\( \\sum_{i<j} x_i x_j = \\dfrac{a_{n-2}}{a_n} \\)
- 三三乘积和：\\( \\sum_{i<j<k} x_i x_j x_k = -\\dfrac{a_{n-3}}{a_n} \\)
- ⋯⋯
- 全部根的乘积：\\( x_1 x_2 \\cdots x_n = (-1)^n \\dfrac{a_0}{a_n} \\)
（规律：符号正负交替，系数比按降幂顺序对应各阶对称和）

### 🔹 为什么重要？
1. **免解方程**：无需求根公式，直接写出根的和与积。
2. **验根与构造**：已知根可快速写出方程；已知部分根可结合定理求未知根。
3. **对称式求值**：处理 \\( x_1^2+x_2^2 \\)、\\( \\frac{1}{x_1}+\\frac{1}{x_2} \\) 等表达式时，通过恒等变形转化为和与积。
4. **跨领域应用**：在函数图像、解析几何、不等式、数学竞赛中频繁出现。

### 🔹 小例子
方程 \\( 2x^2 - 5x + 3 = 0 \\)，由韦达定理：
- \\( x_1 + x_2 = -\\frac{-5}{2} = \\frac{5}{2} \\)
- \\( x_1 x_2 = \\frac{3}{2} \\)
若题目要求 \\( x_1^2 + x_2^2 \\)，可直接计算：
\\( x_1^2 + x_2^2 = (x_1+x_2)^2 - 2x_1x_2 = \\left(\\frac{5}{2}\\right)^2 - 2\\cdot\\frac{3}{2} = \\frac{25}{4} - 3 = \\frac{13}{4} \\)
💡 **注意**：韦达定理在复数域内恒成立，不依赖判别式 Δ 的正负；使用时务必保证最高次项系数 \\( a \\neq 0 \\)。
=====标准====
"""