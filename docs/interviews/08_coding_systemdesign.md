# 卷八 · 通用工程：LeetCode 高频 + 系统设计 · 高频面试题

> 中文具身智能秋招高频面试题库 · **第八卷**
> 题源：CodeTop / 牛客面经汇总 / 一亩三分地 / 知乎秋招手撕合集 / GitHub LeetcodeTop · 系统设计来源见各题
> 同义题合并后 **40 题**（LeetCode 30 + ML 系统设计 5 + 机器人系统设计 5）

**难度分布**：L1（必会） **8** · L2（进阶） **19** · L3（顶级 / 设计） **13**

**使用方式**：题目默认折叠，点开看答案。LeetCode 段（§1）按高频从高到低排，**只给"考察点 / 关键思路 / 易错"**——不贴完整代码；系统设计段（§2-§3）给"答 / 易错"两段，覆盖架构权衡。手机端原生支持。

**与其他卷的关系**：本卷收纳"通用工程能力"题——LeetCode 不挑公司，ML 与机器人系统设计偏架构权衡，与卷一到卷七的"主题知识"正交。**适合岗位**：所有具身 / 自动驾驶 / VLA 算法岗的一面 coding 与 senior 面 system design 阶段。

**说明**：§1 LeetCode 段题号用 `Q01-Q30`，class 为 `qa qa-handcoding`（与各卷 §H 同款）；§2/§3 系统设计段 `Q31-Q40`，class 为 `qa`（视觉上区分两种）。

---

## §1 LeetCode 高频 30 题（按频次排序）

> 来源：CodeTop（字节 / 阿里 / 腾讯 / 美团 / 拼多多等头部互联网 + 自动驾驶车企的近 1 年面经合并统计）。具身 / 自动驾驶岗 coding 一面与互联网算法岗高度重叠。每题给"考察点 / 实现 / 易错"，附 ≤30 行纯 Python（不用 PyTorch）。

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×30</span> <b>Q01</b> · LC3 无重复字符的最长子串</summary>

**考察点**：滑动窗口；哈希记字符最后一次位置；左指针只能右移（用 max 防回退）。

**实现**：

```python
def length_of_longest_substring(s):
    last = {}      # 字符 -> 最后出现下标
    l, ans = 0, 0
    for r, c in enumerate(s):
        # 关键：max(l, last[c]+1)；last[c] 可能已在窗口外，l 不能回退
        if c in last and last[c] >= l:
            l = last[c] + 1
        last[c] = r
        ans = max(ans, r - l + 1)
    return ans
```

**易错**：忘 `max` 致 `l` 回退；用 set 删旧元素逻辑复杂；用 `l = last[c]+1` 不判区间漏过窗口外旧位置。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×25</span> <b>Q02</b> · LC206 反转链表</summary>

**考察点**：三指针迭代 O(1) 空间；递归 O(n) 栈空间；断尾防成环。

**实现**：

```python
def reverse_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next      # 先存下一节点，否则反转后丢失
        cur.next = prev     # 反转当前节点指针
        prev, cur = cur, nxt
    return prev             # prev 为新头

def reverse_recursive(head):
    if not head or not head.next:  # 空或单节点直接返回（base case）
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head   # 让下一节点指回当前
    head.next = None        # 断尾防成环
    return new_head
```

**易错**：忘 `head.next=None` 致成环；指针顺序错（先改 next 再保存致丢失）；递归 base case 漏判 head 空。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×20</span> <b>Q03</b> · LC15 三数之和</summary>

**考察点**：排序 + 双指针；三处去重（i / l / r）；命中后双指针必须同时移动。

**实现**：

```python
def three_sum(nums):
    nums.sort()
    n, res = len(nums), []
    for i in range(n - 2):
        if nums[i] > 0:
            break                       # 已排序，最小值>0 不可能和为 0
        if i > 0 and nums[i] == nums[i - 1]:
            continue                    # ① i 去重：跳过相同首元素
        l, r = i + 1, n - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s == 0:
                res.append([nums[i], nums[l], nums[r]])
                # ②③ l/r 同时去重
                while l < r and nums[l] == nums[l + 1]: l += 1
                while l < r and nums[r] == nums[r - 1]: r -= 1
                l += 1; r -= 1          # 必须同步移动，否则死循环
            elif s < 0:
                l += 1
            else:
                r -= 1
    return res
```

**易错**：忘排序；只对 `i` 去重；命中后只动一侧指针致死循环；漏 `nums[i]>0` 早停。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×18</span> <b>Q04</b> · LC146 LRU 缓存机制</summary>

**考察点**：哈希 + 双向链表 → get/put 都 O(1)；Python `OrderedDict.move_to_end` 一行解决。

**实现**：

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = OrderedDict()  # 内置双向链表，有序

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # 访问后移到尾（最近用）
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            # popitem(last=False) 弹最旧；同步从哈希删（OrderedDict 一并删）
            self.cache.popitem(last=False)
```

**易错**：用单链表无法 O(1) 删中间节点；超容量后只删链表忘删哈希；`get` 已存在 key 忘 `move_to_end`。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×15</span> <b>Q05</b> · LC215 数组中第 K 个最大元素</summary>

**考察点**：小顶堆 O(n log K) vs 快选 O(n) 期望；快选必须随机 pivot 防排序数据退化。

**实现**：

```python
import heapq
import random

def kth_largest_heap(nums, k):
    # 小顶堆维护 top-K：堆顶就是第 K 大
    heap = []
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap[0]

def kth_largest_quickselect(nums, k):
    # 找第 k 大 = 找第 (n-k) 小
    target = len(nums) - k

    def partition(l, r):
        pivot = random.randint(l, r)            # 随机 pivot 防排序数据 O(n²)
        nums[pivot], nums[r] = nums[r], nums[pivot]
        store = l
        for i in range(l, r):
            if nums[i] < nums[r]:
                nums[i], nums[store] = nums[store], nums[i]
                store += 1
        nums[store], nums[r] = nums[r], nums[store]
        return store

    l, r = 0, len(nums) - 1
    while True:
        p = partition(l, r)
        if p == target: return nums[p]
        elif p < target: l = p + 1
        else: r = p - 1
```

**易错**：堆方向反（求第 K 大用小顶堆，反直觉）；快选不随机 pivot 退化；维护全量堆浪费空间。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×14</span> <b>Q06</b> · LC53 最大子序和</summary>

**考察点**：Kadane 算法 O(n)/O(1)；`dp[i]` 定义是"以 i 结尾"而非"前 i 项最大"。

**实现**：

```python
def max_subarray(nums):
    # cur 必须初始化为 nums[0]，不是 0：全负数组时 0 会错
    cur = ans = nums[0]
    for x in nums[1:]:
        # 在线扩展 vs 重启：max(x, cur+x)
        # 等价于 "以 x 结尾的最大子段和"
        cur = max(x, cur + x)
        ans = max(ans, cur)
    return ans
```

**易错**：`cur=0` 在全负数组上返回 0（应 `nums[0]`）；定义记成"前 i 项最大"丢失子串约束；遍历从 0 而非 1 致重复计入 nums[0]。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×12</span> <b>Q07</b> · LC25 K 个一组翻转链表</summary>

**考察点**：dummy 哨兵；不足 K 个保持原序；接回时四个边界指针更新。

**实现**：

```python
def reverse_k_group(head, k):
    dummy = type(head)(0) if head else None
    if not dummy:
        return None
    dummy.next = head
    pre = dummy                          # 上一组的末尾
    while True:
        # 先看后面是否还有 K 个节点；不足 K 直接 break（题目要求保留原序）
        tail = pre
        for _ in range(k):
            tail = tail.next
            if not tail:
                return dummy.next
        nxt_group = tail.next            # 下一组起点，反转前先存
        # 反转 [pre.next, tail] 这一段
        prev, cur = nxt_group, pre.next
        while cur != nxt_group:
            tmp = cur.next
            cur.next = prev
            prev, cur = cur, tmp
        # 接回：原 head 现在是这一组的尾
        new_tail = pre.next
        pre.next = prev                  # 新头接到 pre 后
        pre = new_tail                   # pre 推进到本组末尾
```

**易错**：不足 K 个反了（应保留原序）；`pre` 未更新致后续段错位；递归写法 O(n/K) 栈深大输入炸。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q08</b> · LC121 买卖股票的最佳时机</summary>

**考察点**：一次买卖；一趟扫维护历史最低 + 最大差；变体（多次买卖 / 含冷冻 / 含费）走状态机 DP。

**实现**：

```python
def max_profit(prices):
    if not prices:
        return 0
    # min_price 初始为 prices[0] 而非 inf，保证遍历从 i=1 开始时不错算
    min_price, ans = prices[0], 0
    for p in prices[1:]:
        ans = max(ans, p - min_price)   # 先更答案：不能在同一天卖出
        min_price = min(min_price, p)
    return ans
```

**易错**：把题目当多次买卖；先更新 `min_price` 再算差致同日买卖；DP 写成 O(n²) 双重循环；空数组未判返回错。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×11</span> <b>Q09</b> · LC42 接雨水</summary>

**考察点**：双指针 O(n)/O(1) 最优；挪当前两端柱中较短的那侧（短板锁定本列水位）。

**实现**：

```python
def trap(height):
    # 双指针：比较两端当前高度；较短的一侧的水位由它那侧的 max 锁定
    l, r = 0, len(height) - 1
    lmax, rmax = 0, 0
    ans = 0
    while l < r:
        if height[l] < height[r]:
            # 左边短：左侧 max 比 height[r] 小，水位即 lmax
            lmax = max(lmax, height[l])
            ans += lmax - height[l]
            l += 1
        else:
            # 右边短或相等：水位由 rmax 锁定
            rmax = max(rmax, height[r])
            ans += rmax - height[r]
            r -= 1
    return ans
```

**易错**：挪错方向（应挪当前两端中较短那侧）；单调栈版漏减底高；预处理左右最大值版下标越界。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×11</span> <b>Q10</b> · LC102 二叉树层序遍历</summary>

**考察点**：BFS 按层分组的关键 = 内层用 `n=len(q)` 固定本层数量。

**实现**：

```python
from collections import deque

def level_order(root):
    if not root:
        return []                       # 漏判 None 致后续异常
    res, q = [], deque([root])
    while q:
        n = len(q)                      # 关键：固定本层数量，再分批出队
        level = []
        for _ in range(n):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

**易错**：忘 `n=len(q)` 致分层失败；`root=None` 漏判异常；用 DFS+depth 写也行但不直观。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q11</b> · LC1 两数之和</summary>

**考察点**：一趟哈希；边查边插防自配对；返回下标非数值。

**实现**：

```python
def two_sum(nums, target):
    seen = {}                # val -> idx，已遍历过的数
    for i, x in enumerate(nums):
        need = target - x
        if need in seen:     # 先查再插：防自身和自身配对
            return [seen[need], i]
        seen[x] = i
    return []                # 题目通常保证有解
```

**易错**：预先全塞哈希再查致 `[i,i]` 自配对；返回值是下标但返回了数值；用排序+双指针失下标信息。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q12</b> · LC200 岛屿数量</summary>

**考察点**：网格连通分量；DFS/BFS 任选；原地改 `'0'` 替代 visited 节省空间。

**实现**：

```python
def num_islands(grid):
    if not grid:
        return 0
    R, C = len(grid), len(grid[0])

    def dfs(r, c):
        # 边界 + 已访问/水 三道闸防越界
        if r < 0 or r >= R or c < 0 or c >= C or grid[r][c] != '1':
            return
        grid[r][c] = '0'                # 原地标记访问（题目允许改）
        dfs(r + 1, c); dfs(r - 1, c)
        dfs(r, c + 1); dfs(r, c - 1)    # 四邻居（题目是四邻居）

    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count
```

**易错**：递归无终止条件爆栈；八邻居（题目是四邻居）；不允许改原数组时漏 visited 数组；DFS 大网格用 BFS 更安全。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×9</span> <b>Q13</b> · LC141 环形链表</summary>

**考察点**：Floyd 快慢指针；fast 每步走 2、slow 走 1，有环必相遇。

**实现**：

```python
def has_cycle(head):
    slow = fast = head
    # 边界：fast 与 fast.next 都不能为 None（fast 要走两步）
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def find_cycle_entry(head):
    # LC142 进阶：相遇后 slow 回 head、fast 留原地、同步走 1 步直到相遇
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

**易错**：边界写 `while fast.next` 漏判 `fast=None`；slow 起点写 `head.next` 同步关系乱；用 set 存访问节点占 O(n) 空间。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q14</b> · LC560 和为 K 的子数组</summary>

**考察点**：前缀和差 `s_i - s_j = k`；含负数滑窗不单调必用哈希；`cnt[0]=1` 表空前缀。

**实现**：

```python
from collections import defaultdict

def subarray_sum(nums, k):
    # cnt[前缀和] = 出现次数；初始 cnt[0]=1 让"从头开始就等于 k"被算到
    cnt = defaultdict(int)
    cnt[0] = 1
    s = ans = 0
    for x in nums:
        s += x
        # 先查再插：避免 [i,i] 自配对（含 0 元素时）
        ans += cnt[s - k]
        cnt[s] += 1
    return ans
```

**易错**：含负数用滑窗（窗口和非单调）；漏 `cnt[0]=1` 致全数组和为 k 时漏算；先插再查自配对。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q15</b> · LC236 二叉树最近公共祖先</summary>

**考察点**：递归后序；返回值语义 = "子树里找到 p 或 q 的那个节点"，左右都非空 = 分居两侧 → 当前是 LCA。

**实现**：

```python
def lowest_common_ancestor(root, p, q):
    # base case：root=None 或 root 就是 p/q，直接返回
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    # 左右子树都找到目标 → root 是 LCA
    if left and right:
        return root
    # 否则返回非空那侧（携带 p 或 q 的子树或 LCA 本身）
    return left if left else right
```

**易错**：返回值用 bool 而非节点丢信息；BST 题（LC235）可用大小关系 O(log n)，普通树不行；漏 `root==p or root==q` 早返回。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q16</b> · LC20 有效的括号</summary>

**考察点**：栈匹配；用 dict 存右→左配对；结束栈空才合法。

**实现**：

```python
def is_valid(s):
    pair = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in pair:
            # 遇右括号：栈空或顶不匹配都失败
            if not stack or stack[-1] != pair[c]:
                return False
            stack.pop()
        else:
            stack.append(c)             # 左括号入栈
    return not stack                    # 结束时必须栈空
```

**易错**：栈空时直接 `stack[-1]` 致 IndexError；只判最终栈空但中间错配漏；用计数器无法区分 `()[]` 与 `(]` 的合法性。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×8</span> <b>Q17</b> · LC239 滑动窗口最大值</summary>

**考察点**：单调递减队列；队中存**下标**才能判窗口边界；摊还 O(n)。

**实现**：

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()                        # 存下标，对应值单调递减
    res = []
    for i, x in enumerate(nums):
        # 队头超出窗口：i - k + 1 是窗口左端
        if dq and dq[0] < i - k + 1:
            dq.popleft()
        # 维护单调递减：尾部小于 x 的全弹
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        # i 到达 k-1 才形成第一个完整窗口
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
```

**易错**：deque 存值而非下标 → 无法判窗口边界；窗口左端 `i-k+1` 算错；首次取答案的边界条件 `i>=k-1` 写成 `i>=k`。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q18</b> · LC155 最小栈</summary>

**考察点**：辅助栈与主栈**同步** push/pop（不能只在更小时才压辅助栈）；`getMin` O(1)。

**实现**：

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []             # 与 stack 同步，存历史最小

    def push(self, x):
        self.stack.append(x)
        # 关键：辅助栈每次都压，压的是 min(top, x)；这样 pop 才同步
        m = x if not self.min_stack else min(self.min_stack[-1], x)
        self.min_stack.append(m)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()            # 同步 pop

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]       # O(1)
```

**易错**：辅助栈仅在 `x` 更小时压 → pop 时不同步；`getMin` 用 `min(stack)` 退化 O(n)；忘判空栈。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q19</b> · LC56 合并区间</summary>

**考察点**：按起点排序 + 一次扫描；端点相接（`s<=ce`）也算合并。

**实现**：

```python
def merge(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])  # 按起点排序
    res = [intervals[0][:]]
    for s, e in intervals[1:]:
        # 端点相接也算合并（用 <= 不是 <）
        if s <= res[-1][1]:
            res[-1][1] = max(res[-1][1], e)
        else:
            res.append([s, e])
    return res
```

**易错**：按 end 排序错；`s <= ce` 写成 `<` 漏端点相接；`max(ce, e)` 忘 `max`（新区间可能比旧短）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×7</span> <b>Q20</b> · LC23 合并 K 个升序链表</summary>

**考察点**：小顶堆 O(N log K)；堆元素加 `idx` tie-break 防比较 `node` 报错；分治也行。

**实现**：

```python
import heapq

def merge_k_lists(lists):
    # 寻找任一非空节点拿到节点类型，构造 dummy；只看 lists[0] 会漏 [None, node] 情况
    sample = next((h for h in lists if h), None)
    if sample is None:
        return None
    dummy = type(sample)(0)
    heap = []
    # 所有头节点入堆；(val, idx, node)：idx 是 tie-break 防 val 相同时比较 node 报错
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))
    cur = dummy
    counter = len(lists)
    while heap:
        val, _, node = heapq.heappop(heap)
        cur.next = node
        cur = node
        if node.next:
            heapq.heappush(heap, (node.next.val, counter, node.next))
            counter += 1
    return dummy.next
```

**易错**：堆元素 `(val, node)` val 相等时比较 node 报 TypeError；直接 K 路合并 O(N·K) 而非 O(N log K)；分治区间边界算错。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q21</b> · LC300 最长递增子序列</summary>

**考察点**：tails 二分 O(n log n)；tails 不是答案序列但长度=LIS 长度；严格 vs 非严格选 `bisect_left/right`。

**实现**：

```python
import bisect

def length_of_lis(nums):
    # tails[k] = 所有长度 k+1 LIS 中末尾的最小值；自然单调递增
    tails = []
    for x in nums:
        # 严格递增用 bisect_left（替换第一个 >= x 的位置）
        # 若题目要求非严格递增则用 bisect_right
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)             # x 比所有尾部都大，扩展长度
        else:
            tails[i] = x                # 否则更新让该长度的尾部更小
    return len(tails)

def length_of_lis_dp(nums):
    # O(n²) 经典 DP，便于讲解
    if not nums:
        return 0
    dp = [1] * len(nums)                # 每个元素自己就是长度 1
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

**易错**：把 `tails` 直接当 LIS 输出（错；只长度对）；严格/非严格用错 `bisect_left/right`；DP 初值漏 1 致全 0。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q22</b> · LC994 腐烂的橘子</summary>

**考察点**：多源 BFS（所有腐烂橘子同时入队）；初始层不计时间；`fresh>0` 返回 -1。

**实现**：

```python
from collections import deque

def oranges_rotting(grid):
    R, C = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    # 多源 BFS：所有腐烂橘子先入队
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2:
                q.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1
    if fresh == 0:
        return 0                        # 没新鲜橘子直接 0 分钟
    minutes = 0
    while q and fresh > 0:
        for _ in range(len(q)):
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 1:
                    grid[nr][nc] = 2    # 标记腐烂
                    fresh -= 1
                    q.append((nr, nc))
        minutes += 1                    # 一轮 BFS 后才 +1（初始层不计）
    return minutes if fresh == 0 else -1
```

**易错**：单源 BFS 漏远处腐烂源；`minutes` 多算一层；`fresh==0` 时未直接返回 0 进入 BFS 多算 0→1。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q23</b> · LC32 最长有效括号</summary>

**考察点**：栈底放 `-1` 哨兵，`i - stack[-1]` 算长度；'(' 入栈、')' 弹栈后栈空则用 `i` 作新哨兵。

**实现**：

```python
def longest_valid_parentheses(s):
    stack = [-1]                        # 哨兵：例如 "()" 在 i=1 处弹出后，长度 = 1-(-1)=2 正确
    ans = 0
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        else:                           # ')'
            stack.pop()
            if not stack:
                # 栈空说明当前 ')' 多余，用 i 作新哨兵
                stack.append(i)
            else:
                # 当前合法段长度 = i - 上一未匹配位置
                ans = max(ans, i - stack[-1])
    return ans
```

**易错**：忘哨兵 `-1` 致首段长度算错；')' 弹栈后忘判空致后续算法挂；用计数器无法处理 `()(()` 这类嵌套+并列。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q24</b> · LC72 编辑距离</summary>

**考察点**：二维 DP；三种操作（删 / 增 / 改）对应 `dp[i-1][j]`、`dp[i][j-1]`、`dp[i-1][j-1]`；边界 `dp[0][j]=j、dp[i][0]=i`。

**实现**：

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    # dp[i][j] = word1[:i] -> word2[:j] 的最少操作数
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i                    # 删 i 个变空串
    for j in range(n + 1):
        dp[0][j] = j                    # 插 j 个变 word2[:j]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 删 word1[i-1] / 在 word1 插 word2[j-1] / 改字符
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
```

**易错**：下标 1 偏移（`word1[i-1]`）；`min` 三项漏一项；空间优化时左上角 `dp[i-1][j-1]` 被覆盖前忘暂存。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q25</b> · LC5 最长回文子串</summary>

**考察点**：中心扩展 O(n²)；每个位置同时考虑奇中心（i,i）与偶中心（i,i+1）。

**实现**：

```python
def longest_palindrome(s):
    if not s:
        return ""

    def expand(l, r):
        # 从中心向两侧扩展，返回最长回文的 (start, length)
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return l + 1, r - l - 1         # 退出时 l/r 已越过端点

    start, max_len = 0, 1
    for i in range(len(s)):
        s1, l1 = expand(i, i)           # 奇中心
        s2, l2 = expand(i, i + 1)       # 偶中心（必算，不能漏）
        if l1 > max_len:
            start, max_len = s1, l1
        if l2 > max_len:
            start, max_len = s2, l2
    return s[start:start + max_len]
```

**易错**：只算奇中心漏偶；扩展时未判 `l>=0 and r<n` 越界；返回时起点/长度计算错误。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q26</b> · LC46 全排列</summary>

**考察点**：回溯模板；`used[]` 数组标已用；进出递归时成对地"做选择/撤销选择"。

**实现**：

```python
def permute(nums):
    n = len(nums)
    res = []
    used = [False] * n
    path = []

    def dfs():
        if len(path) == n:
            res.append(path[:])         # 收集时必须拷贝快照
            return
        for i in range(n):
            if used[i]:
                continue
            # 做选择
            used[i] = True
            path.append(nums[i])
            dfs()
            # 撤销选择（成对出现）
            path.pop()
            used[i] = False

    dfs()
    return res
```

**易错**：收集 `res.append(path)` 而非 `path[:]` 致引用共享被改污染；忘撤销 `used[i]=False`；LC47 去重时条件方向反（应在 `not used[i-1]` 时跳过同层重复枝）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q27</b> · LC19 删除链表倒数第 N 个节点</summary>

**考察点**：快慢指针一次遍历；dummy 防删头节点；`fast` 先走 n 步后再与 `slow` 同步。

**实现**：

```python
def remove_nth_from_end(head, n):
    # dummy 解决"待删的是头节点"的边界
    dummy = type(head)(0)
    dummy.next = head
    slow = fast = dummy
    # fast 先走 n 步；之后 slow 与 fast 同步走到 fast.next 为 None
    # 此时 slow 恰好在待删节点的前一个
    for _ in range(n):
        fast = fast.next
    while fast.next:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next          # 跳过待删节点
    return dummy.next
```

**易错**：不用 dummy 时删 head 失败；`fast` 走 n+1 步致 slow 停错位（取决于循环写法）；`fast.next` 判空写成 `fast` 致越界。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q28</b> · LC22 括号生成</summary>

**考察点**：回溯剪枝；`left<n` 才加 '('、`right<left` 才加 ')'；两 if 都要尝试（不是 elif）。

**实现**：

```python
def generate_parenthesis(n):
    res = []

    def dfs(path, left, right):
        if len(path) == 2 * n:
            res.append(''.join(path))
            return
        if left < n:                    # 还能加左括号
            path.append('(')
            dfs(path, left + 1, right)
            path.pop()
        if right < left:                # 关键：right<left 才合法（不是 <=）
            path.append(')')
            dfs(path, left, right + 1)
            path.pop()

    dfs([], 0, 0)
    return res
```

**易错**：剪枝写 `right<=left` 致 `())` 类非法解漏过；用 `elif` 替代两 if 致漏解；忘 `path.pop` 回溯。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q29</b> · LC1143 最长公共子序列</summary>

**考察点**：二维 DP；子序列（可不连续）与子串（必连续）转移区别——子串错位时 `dp[i][j]=0`，子序列取 `max(dp[i-1][j], dp[i][j-1])`。

**实现**：

```python
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    # dp[i][j] = text1[:i] 与 text2[:j] 的 LCS 长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # 字符匹配：继承左上 + 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 不匹配：放弃 text1[i-1] 或 text2[j-1]，取大
                # 注意：子串（非子序列）这里应是 0，转移方式完全不同
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**易错**：与最长公共子串混淆（子串错位清零）；下标 1 偏移；`max` 漏一项。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q30</b> · LC169 多数元素</summary>

**考察点**：Boyer-Moore 投票；多数元素（出现 > n/2）必能胜出；O(n)/O(1)。

**实现**：

```python
def majority_element(nums):
    # cand: 当前候选；count: 净计数
    cand, count = None, 0
    for x in nums:
        if count == 0:
            cand = x                    # 重置候选
            count = 1
        elif x == cand:
            count += 1
        else:
            count -= 1                  # 互相抵消
    # 题目保证存在多数元素；不保证时再扫一遍验证 count(cand) > n/2
    return cand
```

**易错**：把多数元素当"出现最多"（不一定 >n/2）；`count=0` 时不重置 `cand` 致结果错；LC229（>n/3）需要两个候选。

</details>

---

## §2 ML 系统设计（5 题）

> ML 平台 / 训练基建 / 推理服务的架构题，2 面 / 3 面 senior 阶段常考。来源：知乎 OpenVLA / π0 综述、vLLM / SGLang 文档、HF RLHF 系列博客。

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×5</span> <b>Q31</b> · 设计一个 VLA 训练 pipeline（数据 → 训练 → 部署）</summary>

**答**：

**数据层**：RLDS / LeRobot HF dataset（state, action, image, language）；cross-embodiment 按 action space 对齐（dof / 末端 vs 关节）；按 demo 成功率与轨迹平滑度过滤。

**训练层**：视觉 backbone（SigLIP / DINOv2 / CLIP）+ 主干 LLM（OpenVLA = LLaMA-2-7B、π0 = PaliGemma-3B）；DDP + ZeRO-2 / FSDP（>7B 必上）；warmup + cosine、bf16；EMA 推理权重。

**部署层**：量化（FP16 / int8）、TensorRT / vLLM、action chunking 解耦推理与控制频率、A/B + rollback。

**易错**：不同 dof 需 action token 化或多 head；mixed precision 不开浪费显存；data loader 是 bottleneck 没 prefetch。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×4</span> <b>Q32</b> · 设计 VLA 推理服务（多机器人并发调用一个 endpoint）</summary>

**答**：

**Batching**：动态 batch（continuous batching, vLLM 风格）；prefill 与 decode 分别 batch。

**KV cache**：PagedAttention 思想避免内存碎片；多请求共享 system prompt 复用 prefix cache。

**频率解耦**：控制环 50-1000 Hz，VLA 推理 5-25 Hz → action chunking（一次预测 H 步动作，底层控制器以高频插值）。

**多模型路由**：按任务类型 / embodiment 分流到不同 backbone；统一 API。

**SLO**：p50 / p99 延迟、QPS、actions/sec；监控 OOM、cache hit rate。

**易错**：用同步 batch 等慢请求致控制环抖；KV cache 跨请求不共享浪费显存；忽略 streaming（首 token 即可触发下游）。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q33</b> · 设计 RLHF / preference 数据采集系统</summary>

**答**：

**采集 UI**：pairwise comparison 界面，展示同 prompt 下两条 response 选偏好；可选"难以判断"档。

**Schema**：`(prompt_id, response_a, response_b, choice, annotator_id, ts, confidence)`；prompt / response 单独去重表。

**一致性**：多标注员（≥3）对同一对判断算 Fleiss' κ 或 Krippendorff's α（Cohen's κ 仅适合 2 人对比）；κ<0.4 舍弃或重标；定期注入 gold sample 校准。

**RM 训练**：解耦数据与训练；周期 dump → 训新 RM → A/B 上线 → 版本控制。

**隐私**：脱敏 PII；annotator 看不到 user_id；prompt 含敏感词过滤。

**易错**：不校验 annotator bias；RM 用全部数据无 hold-out 致过拟合；忘存 confidence 字段。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q34</b> · 设计 multi-robot fleet 数据采集系统</summary>

**答**：

**异构汇总**：按 RLDS / OpenX-Embodiment 统一 schema；保 `embodiment_id` 便于按机型筛。

**Action 标准化**：delta vs absolute 各自归一化；末端用 SE(3)；joint space 按 dof 切分。

**时间同步**：硬件 PTP / hardware trigger 优于软件时间戳；每帧打 `(robot_ts, server_ts)` 双时间戳。

**传感器对齐**：相机内外参标定文件入 metadata；点云时间戳对齐到 RGB。

**隐私**：人脸 / 车牌 blur；GPS 改为室内相对坐标。

**存储**：原始冷存 S3 / OSS；训练用切片走 WebDataset / Parquet。

**易错**：不同控制频率混致 action chunk 长度不一；忘 calibration 跨 robot 不可用；时间戳跨节点偏差需 NTP / PTP。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q35</b> · 设计 LLM offline batch inference for trajectory labeling</summary>

**答**：

**框架**：vLLM / SGLang 做批量推理；DeepSpeed-MII 备选。

**Prefix cache**：同一 system prompt 跨样本共享 KV cache（模板固定、user input 变）→ 吞吐 2-5×。

**Sharding**：数据按 GPU 数切片；调 `vllm.LLM.generate(prompts, sampling_params)` 批跑。

**失败 / 重试**：每 100 样本写 checkpoint；挂了从 checkpoint 恢复；OOM 时降 batch 重试。

**质量监控**：抽样 N=100 人工检查；与小模型答案对比检测 outlier。

**Cost**：input + output tokens × 单价；用 `tiktoken` 预估。

**易错**：没用 prefix cache 浪费算力；OOM 后没自动降 batch；output 没 streaming 写盘致跑完才存。

</details>

---

## §3 机器人系统设计（5 题）

> 机器人整栈架构题，覆盖 perception / planning / control / 多传感器 / SLAM。来源：知乎 ROS 具身入门 / 自动驾驶规控面经 / 深蓝学院 ROS 课程。

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×4</span> <b>Q36</b> · 设计 perception-planning-control 完整栈</summary>

**答**：

**模块（频率）**：
- Perception：LiDAR / camera fusion（10-30 Hz）。
- State Estimation：IMU + 编码器 + VO（100-200 Hz）。
- Planning：global A* / RRT（1-5 Hz）+ local DWA / MPC（10-50 Hz）。
- Control：PID / MPC / impedance（100-1000 Hz）。

**通信**：ROS 2 + DDS；topic / service / action 按场景选。

**安全层**：emergency stop（硬+软双重）；collision check 在 planning 前；watchdog 心跳。

**Sim-to-Real**：模块单测；Isaac Lab / MuJoCo regression；真机灰度。

**易错**：所有模块同频运行浪费算力；忘 safety layer 出事故；perception 直送 control 缺 state estimation 中间层。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q37</b> · 设计 multi-sensor fusion 系统（camera + LiDAR + IMU）</summary>

**答**：

**时间同步**：hardware trigger（PTP / GPS PPS）μs 级，远优于软件时间戳（ms 级）；车端常用 GMSL 相机硬触发。

**标定**：内参（焦距、畸变）+ 外参（相机-LiDAR-IMU SE(3) 变换）；用 Kalibr / lidar_align 工具链。

**融合算法**：
- EKF / UKF：状态 = pose + vel + IMU bias；VIO 主流，适实时。
- Factor graph（GTSAM / Ceres）：批量优化，精度高但延迟大。

**故障处理**：sensor health monitor 检测 dropout / outlier；降级模式（如 LiDAR 失效退回纯视觉 VIO）。

**Covariance**：由传感器规格表 + Allan variance 标定，不要随手填。

**易错**：用软同步致 BA 漂移；忘 IMU 预积分；故障检测漏致错误数据污染 fusion。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q38</b> · 设计 VLN agent 系统（自然语言指令 → 导航）</summary>

**答**：

**语义地图**：边走边用 CLIP / Grounding DINO 标 voxel grid 或 BEV map，把"红色椅子"对应到地图坐标。

**指令解析**：LLM 拆 subgoal（"先到沙发再到厨房"→ list of waypoint queries）；CoT 提取空间关系。

**Waypoint policy**：每步预测 next waypoint 像素坐标，反投影到 3D；NaVid / NaVILA / VLN-R1 类 VLM 推理。

**Low-level**：waypoint → A* / RRT → 局部 DWA → motor cmd。

**Failure recovery**：卡死检测（pose 不变 > T s）；重规划或回到上一 waypoint；ask-for-help。

**Closed-loop**：每 1-2 s 重新评估当前 vs goal，不开环跑完。

**易错**：open-loop 跑完才发现卡死；hallucination 无置信度过滤；LLM 调用频率过高致延迟爆掉。

</details>

<details class="qa">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×3</span> <b>Q39</b> · 设计 SLAM 状态估计模块</summary>

**答**：

**Frontend**：特征提取（ORB / SuperPoint）+ 匹配（FLANN / SuperGlue）+ 相对位姿求解（PnP / 5-point + RANSAC）。

**Backend**：因子图 / 位姿图优化（GTSAM / g2o / Ceres）；BA 联合优化相机 pose + 地图点；滑窗 BA 控制规模。

**IMU**：预积分把测量打包成相对位姿增量，作为因子加入因子图（VIO 主流）。

**Loop closure**：DBoW / NetVLAD 词袋检测候选；几何校验（PnP RANSAC）+ pose graph 优化纠正累积漂移。

**鲁棒性**：动态物体过滤（语义 / 运动一致性）；初始化判稳；故障时切 odometry-only 模式。

**易错**：frontend / backend 划分不清；忘 loop closure 致长轨迹漂移；IMU 没预积分（重积分巨慢）；动态物体不过滤致 ego-motion 错乱。

</details>

<details class="qa">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×3</span> <b>Q40</b> · 设计 ROS 2 节点架构（双臂操作 demo）</summary>

**答**：

**节点拆分**：Perception（相机驱动 + 6D pose） / Planning（MoveIt 2 双臂） / Arm_controller × 2 + Gripper_controller / Coordinator（状态机或 behavior tree 编排）。

**通信**：topic（高频 telemetry）；service（同步如 `compute_ik`）；action（长时任务 `move_to_pose`，含 feedback / cancel）。

**QoS**：sensor 用 `BEST_EFFORT + KEEP_LAST(10)`；控制指令用 `RELIABLE + KEEP_LAST(1)`。

**Launch**：分层（hardware / perception / planning / app），便于真机 / 仿真切换。

**易错**：所有功能塞一个节点；长时任务用 service 阻塞主循环（应 action）；QoS 默认全 reliable 致传感器 topic 卡顿。

</details>
