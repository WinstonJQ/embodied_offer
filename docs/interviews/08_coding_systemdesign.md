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

> 来源：CodeTop（字节 / 阿里 / 腾讯 / 美团 / 拼多多等头部互联网 + 自动驾驶车企的近 1 年面经合并统计）。具身 / 自动驾驶岗 coding 一面与互联网算法岗高度重叠——这 30 题覆盖了链表 / 滑动窗口 / DP / 树图 / 栈 / 设计的主流分布。

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×30</span> <b>Q01</b> · LC3 无重复字符的最长子串</summary>

**考察点**：滑动窗口模板；哈希记位置；窗口左端怎么跳。

**关键思路**：
- 用 `dict` 记每个字符最后一次出现的下标。
- 右指针 r 遍历；遇到重复字符时，左指针 l 跳到 `max(l, last[c] + 1)`，注意 max（旧位置可能已在窗口外）。
- 答案 = `max(ans, r - l + 1)`。
- 复杂度 O(n)，空间 O(min(n, |Σ|))。

**易错**：忘 max 导致 l 回退；用 set 删除老元素逻辑变复杂；初始 l=0 但首字符就重复时 +1 偏移。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×25</span> <b>Q02</b> · LC206 反转链表</summary>

**考察点**：单链表三指针迭代；递归写法。

**关键思路**：
- 迭代：`prev=None, cur=head`，循环 `nxt=cur.next; cur.next=prev; prev=cur; cur=nxt`，返回 prev。
- 递归：`new_head = reverse(head.next); head.next.next = head; head.next = None`，base case `not head or not head.next`。
- 复杂度 O(n)；迭代空间 O(1)，递归 O(n)。

**易错**：忘断尾 `head.next=None` 致环；指针顺序错先改 next 再保存致丢失；递归 base case 漏判 head 空。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×20</span> <b>Q03</b> · LC15 三数之和</summary>

**考察点**：排序 + 双指针；如何去重。

**关键思路**：
- 先排序 O(n log n)。
- 外层固定 `i`，内层 `l = i+1, r = n-1` 双指针；和 < 0 左移、> 0 右移、= 0 入答案。
- 去重三处：① `i > 0 且 nums[i] == nums[i-1]` 跳；② 命中后 `while nums[l] == nums[l+1]: l++`；③ `r` 同理。
- 复杂度 O(n²)。

**易错**：忘排序；只对 i 去重忘了 l/r 去重；左右指针没同时移动导致死循环。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×18</span> <b>Q04</b> · LC146 LRU 缓存机制</summary>

**考察点**：哈希 + 双向链表组合；get / put 都要 O(1)。

**关键思路**：
- 哈希表 key → node；双向链表按访问顺序排（最近用的放头部）。
- get：哈希查 node → 移到头部 → 返回值。
- put：若 key 存在更新值并移头；否则新建插头，超容量删尾节点（同步删哈希）。
- Python 可用 `OrderedDict` + `move_to_end`；C++ 用 list + unordered_map。

**易错**：单链表无法 O(1) 删除中间节点；哈希值存数值而非节点指针；超容量后忘从哈希删 key。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×15</span> <b>Q05</b> · LC215 数组中第 K 个最大元素</summary>

**考察点**：堆 vs 快速选择；面试官常追问 O(n) 平均做法。

**关键思路**：
- 小顶堆维护 top-K：堆大小 > K 时弹堆顶；最后堆顶即第 K 大。复杂度 O(n log K)。
- 快速选择（QuickSelect）：随机 pivot 分区，递归到目标分区。期望 O(n)，最差 O(n²) 需随机化避免。
- Python：`heapq.nlargest(k, nums)[-1]` 直接给答案。

**易错**：大顶堆维护全量浪费空间；QuickSelect 不随机选 pivot 在排序数据上退化；混淆"第 K 大"与"第 K 小"导致堆方向反。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×14</span> <b>Q06</b> · LC53 最大子序和</summary>

**考察点**：一维 DP；Kadane 算法（在线扩展 vs 重启）。

**关键思路**：
- `dp[i] = max(nums[i], dp[i-1] + nums[i])`，答案 = `max(dp)`。
- 空间优化为单变量 `cur = max(x, cur + x)`，O(1) 空间。
- 进阶：分治 O(n log n)（线段树思路），返回 `(总和, 最大前缀, 最大后缀, 最大子段)`。

**易错**：dp[i] 表示"以 i 结尾"而不是"前 i 项最大"，定义记错；初始值 cur=0 在全负数组上得 0（应初始化 cur = nums[0]）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×12</span> <b>Q07</b> · LC25 K 个一组翻转链表</summary>

**考察点**：分组反转的边界处理；不足 K 个保持原序。

**关键思路**：
- 先 dummy 哨兵；遍历找到下一段 K 个节点（不足 K 则 break）。
- 反转这 K 个节点：复用 reverse 模板，传入 `head, tail` 边界。
- 接回：原 `pre.next = new_head`，原 `head.next = next_group`，更新 pre = head。
- 复杂度 O(n)。

**易错**：不足 K 个反了（题目要求保留）；接回时 pre 没正确更新致后续段错位；递归写法栈深 O(n/K) 大输入炸栈。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×12</span> <b>Q08</b> · LC121 买卖股票的最佳时机</summary>

**考察点**：一次买卖（只能买卖一次）的最优答案；扫一遍维护最小值。

**关键思路**：
- 扫描数组，维护 `min_price = min(min_price, p)`。
- 答案 = `max(ans, p - min_price)`。
- 复杂度 O(n) / 空间 O(1)。
- 变体（多次买卖 LC122）：贪心累加所有正差；含冷冻期 LC309 / 含手续费 LC714 用状态机 DP。

**易错**：把题目当多次买卖；min_price 初始化为 inf 而非 prices[0]（数组只有一个元素时返回 0）；DP 写成 O(n²)（外双重循环）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×11</span> <b>Q09</b> · LC42 接雨水</summary>

**考察点**：双指针 vs 单调栈 vs 预处理左右最大值；空间复杂度 trade-off。

**关键思路**：
- 双指针（最优）：`l, r` 双端；`lmax, rmax` 记两侧最大；哪边 max 小哪边动，移动时加 `lmax - h[l]` 或 `rmax - h[r]`。O(n) 时间 / O(1) 空间。
- 单调栈：栈存下标，碰到更高柱子结算"凹槽水量"。
- 预处理：左右最大值各扫一遍，对每个 i 加 `min(L[i], R[i]) - h[i]`。

**易错**：双指针挪错方向（应挪 max 较小那侧）；单调栈算高度时忘减底 `h[stack[-1]]`；左右数组下标越界。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×11</span> <b>Q10</b> · LC102 二叉树层序遍历</summary>

**考察点**：BFS 模板；如何按层分组（外循环逐层）。

**关键思路**：
- `deque` 初始入 root；外层 `while q`，记录 `n = len(q)`；内层取 n 个节点（即当前层）。
- 每取一个节点出队，左右孩子入队（非空）。
- 收集本层值到 `level_vals`，加入 `result`。
- 复杂度 O(n)，空间 O(w) w 为最宽层。

**易错**：忘记内层用 `n = len(q)` 固定本层数量（直接 while q 不能分层）；root=None 漏判直接异常；DFS 递归靠 depth 参数也能写但不直观。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×10</span> <b>Q11</b> · LC1 两数之和</summary>

**考察点**：一趟哈希；为何不用排序+双指针（题目要求返回下标）。

**关键思路**：
- 哈希表 `{val: idx}`，遍历 nums。
- 当前 x，查 `target - x` 是否已在哈希；在则返回 `[d[target-x], i]`。
- 否则 `d[x] = i` 加入哈希。
- 复杂度 O(n) / 空间 O(n)。

**易错**：先把全部数值塞哈希再查会出现 `[i, i]` 自配对（应边查边插）；返回值是下标不是数值。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×10</span> <b>Q12</b> · LC200 岛屿数量</summary>

**考察点**：网格连通分量；DFS / BFS 任选；标记访问的两种方式。

**关键思路**：
- 双重循环遍历，遇到 `'1'` 计数 +1 并 DFS / BFS 把相连的 1 全标记为 `'0'`（原地）或加入 visited。
- 四邻居方向数组 `dx=[-1,1,0,0]; dy=[0,0,-1,1]`。
- 递归 DFS 注意栈深；大网格用 BFS 更安全。
- 复杂度 O(M·N)。

**易错**：递归无终止条件爆栈；改原数组但题目不允许（拷贝一份）；八邻居（题目是四邻居）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×9</span> <b>Q13</b> · LC141 环形链表</summary>

**考察点**：Floyd 快慢指针；为何一定能相遇。

**关键思路**：
- `slow, fast = head, head`；`while fast and fast.next: slow=slow.next; fast=fast.next.next; if slow==fast: return True`。
- 若 fast 走到 None 返回 False。
- 进阶 LC142 找环入口：相遇后 slow 回 head、fast 留原地，同步走一步直到相遇。

**易错**：fast 边界判断写成 `while fast.next` 漏掉 fast 自己是 None；慢指针起点 head.next 也能写但要小心同步关系。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q14</b> · LC560 和为 K 的子数组</summary>

**考察点**：前缀和 + 哈希；为何不能滑窗（含负数）。

**关键思路**：
- 维护前缀和 `s`；哈希 `cnt[s] = 出现次数`，初始 `cnt[0]=1`（空前缀）。
- 遍历到位置 i 时，`ans += cnt[s - k]`，再把 `s` 入哈希。
- 复杂度 O(n)。

**易错**：含负数不能用滑窗（窗口和非单调）；忘初始化 `cnt[0]=1`，全数组和为 k 时漏算；先更新哈希再查会自配对。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×8</span> <b>Q15</b> · LC236 二叉树最近公共祖先</summary>

**考察点**：递归后序；返回值的语义（找到了什么）。

**关键思路**：
- 函数返回：当前子树中找到 p 或 q 的那个节点；都没找到返回 None。
- base case：`root is None / root == p / root == q` → 返回 root。
- 递归左右子树；左右都非空说明分居两侧 → 当前 root 是 LCA；否则返回非空那侧。
- 复杂度 O(n)。

**易错**：把"找到 p 或 q"理解成"找到 LCA"；递归返回值用 bool 而非节点丢信息；二叉搜索树（LC235）可用 BST 性质 O(log n)，普通二叉树不行。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×8</span> <b>Q16</b> · LC20 有效的括号</summary>

**考察点**：栈匹配；map 存配对关系。

**关键思路**：
- 维护栈和配对 dict `{')':'(', ']':'[', '}':'{'}`。
- 遇左括号入栈；遇右括号检查栈顶是否匹配，不匹配返回 False，匹配则弹栈。
- 结束时栈空才是合法。
- 复杂度 O(n)。

**易错**：栈空时碰到右括号没判空导致 IndexError；只检查最终栈空但中间没匹配；用计数器（不能区分 () 与 [] 的嵌套关系）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×8</span> <b>Q17</b> · LC239 滑动窗口最大值</summary>

**考察点**：单调队列；窗口左端怎么及时弹出。

**关键思路**：
- 用 deque 存**下标**，保证队中对应值单调递减。
- 新元素入队前从尾部弹出比它小的值。
- 队头下标若 `< i - k + 1` 表示已出窗口，弹出。
- 当 i ≥ k-1 时，记录 `nums[queue[0]]` 为当前窗口最大。

**易错**：deque 存值而非下标导致无法判断"何时滑出窗口"；窗口长度边界 `i - k + 1` 算错；遍历完忘补尾部窗口结果。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×7</span> <b>Q18</b> · LC155 最小栈</summary>

**考察点**：辅助栈维护当前最小值。

**关键思路**：
- 主栈正常 push / pop。
- 辅助栈：push 时压入 `min(辅助栈顶, x)`（栈空则压 x）；pop 时辅助栈同步 pop。
- `getMin` 直接返回辅助栈顶，O(1)。
- 进阶：只用一个栈存 `(val, cur_min)` pair。

**易错**：辅助栈只在 x 更小时压入（导致 pop 时不同步）；getMin 用 min(stack) 退化为 O(n)；忘了空栈检查。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q19</b> · LC56 合并区间</summary>

**考察点**：按起点排序 + 一次扫描合并。

**关键思路**：
- 按 `start` 排序。
- 维护当前合并区间 `[cs, ce]`；下一个 `[s, e]`：若 `s <= ce` 则 `ce = max(ce, e)` 合并；否则把 `[cs, ce]` 入答案、开新区间。
- 复杂度 O(n log n)。

**易错**：按 end 排序错；`s <= ce` 写成 `<` 漏端点相接情况（题目通常算合并）；忘了循环结束后把最后一个区间入答案。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×7</span> <b>Q20</b> · LC23 合并 K 个升序链表</summary>

**考察点**：堆 vs 分治；时间复杂度证明。

**关键思路**：
- 堆：所有头节点入小顶堆（按值），弹堆顶接到结果尾、堆顶 next 再入堆。复杂度 O(N log K)，N 为总节点数。
- 分治：两两合并，logK 层，每层 O(N) → O(N log K)。
- Python heap 元素是 `(val, idx, node)`，idx 防 val 相等时比较 node 报错。

**易错**：直接合并 K 路得 O(N·K)；堆比较 node 报错（需加 tie-break 项）；分治合并区间边界算错。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×7</span> <b>Q21</b> · LC300 最长递增子序列</summary>

**考察点**：DP O(n²) 经典写法 + 二分 O(n log n) 进阶。

**关键思路**：
- DP O(n²)：`dp[i] = max(dp[j]) + 1`，j < i 且 `nums[j] < nums[i]`；答案 = max(dp)。
- O(n log n)：维护 tails 数组（不严格是真正 LIS）；新元素若大于尾部则 append、否则二分替换第一个 ≥ 它的位置。tails 长度 = LIS 长度。
- 二分版需理解：tails 不是答案序列，但其长度等于 LIS 长度。

**易错**：把 tails 直接当 LIS（错）；严格 vs 非严格递增（用 bisect_left vs bisect_right）；DP 初始值漏 1。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q22</b> · LC994 腐烂的橘子</summary>

**考察点**：多源 BFS；同时从多个起点扩散。

**关键思路**：
- 初始扫一遍，所有腐烂橘子下标入队，统计新鲜橘子数 `fresh`。
- BFS 逐层；每层结束 `minutes += 1`；每"感染"一个新鲜橘子 fresh -= 1。
- 终止：队空。若 fresh > 0 返回 -1，否则返回 minutes（注意初始层不算时间）。
- 复杂度 O(MN)。

**易错**：BFS 终止时 minutes 多算一层（初始入队的不算新感染）；fresh==0 时直接返回 0，不进 BFS；忘了多源（单源 BFS 会漏掉早就在远处的腐烂源）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q23</b> · LC32 最长有效括号</summary>

**考察点**：栈 vs DP 两种解法。

**关键思路**：
- 栈：栈底放 -1 哨兵；遇 '(' 压下标；遇 ')' 弹栈，若栈空则压当前 i 当新哨兵，否则 `ans = max(ans, i - stack[-1])`。
- DP：`dp[i]` 表"以 i 结尾的最长有效"；只在 s[i]==')' 时算：若 s[i-1]=='(' 则 `dp[i] = dp[i-2] + 2`；若 s[i-1]==')' 且 s[i - dp[i-1] - 1]=='(' 则配对再加前段。
- 复杂度 O(n)。

**易错**：栈底没放 -1 哨兵导致首段长度算错；DP 状态转移条件漏 case；用计数器（不能区分嵌套与并列）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l3">L3</span> <span class="freq">🔥×6</span> <b>Q24</b> · LC72 编辑距离</summary>

**考察点**：二维 DP；三种操作的状态转移。

**关键思路**：
- `dp[i][j]` = word1[:i] 转成 word2[:j] 的最少操作数。
- 边界：`dp[0][j]=j, dp[i][0]=i`。
- 转移：`w1[i-1]==w2[j-1]` 时 `dp[i][j]=dp[i-1][j-1]`；否则 `1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])`（删/增/改）。
- O(mn)；可滚动数组到 O(min(m,n))。

**易错**：下标 1 偏移；min 三项漏一项；空间优化时左上角 `dp[i-1][j-1]` 被覆盖前要先存。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q25</b> · LC5 最长回文子串</summary>

**考察点**：中心扩展 vs Manacher；对奇偶长度的处理。

**关键思路**：
- 中心扩展 O(n²)：对每个 i，分别以 i 为中心（奇）和 (i, i+1) 为中心（偶）向两侧扩展。
- 记录最大长度 + 起点。
- Manacher O(n)：在字符间插 '#' 把奇偶统一，再用对称性加速。
- DP O(n²) 也可（空间 O(n²) 略亏）。

**易错**：只处理奇数中心漏偶数；扩展时边界越界没判；返回起点和长度算错。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q26</b> · LC46 全排列</summary>

**考察点**：回溯模板；used 数组 vs 交换法。

**关键思路**：
- DFS 带 used[] 标记已用：当前 path 长度 == n 时收集；否则枚举未用元素递归。
- 交换法：固定前 i 个，依次把后面元素与 i 交换、递归 i+1、再换回（恢复）。
- 复杂度 O(n!)；含重复元素（LC47）需排序 + 同层去重：`if i>0 and nums[i]==nums[i-1] and not used[i-1]: skip`。

**易错**：忘 used 回滚；交换法没换回致结果污染；LC47 去重条件方向反（应在 `not used[i-1]` 时跳过同层重复枝）。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×6</span> <b>Q27</b> · LC19 删除链表倒数第 N 个节点</summary>

**考察点**：快慢指针一次遍历；dummy 哨兵防删头。

**关键思路**：
- dummy → head；slow=fast=dummy。
- fast 先走 n+1 步；之后 slow、fast 同步走，fast 到 None 时 slow 在待删节点的前一个。
- `slow.next = slow.next.next`；返回 `dummy.next`。
- 复杂度 O(L) 一次遍历。

**易错**：不要 dummy 时删 head 失败；fast 先走 n 步（应 n+1）致 slow 停错；fast.next 而非 fast 判 None 致越界。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q28</b> · LC22 括号生成</summary>

**考察点**：回溯剪枝；左右括号数约束。

**关键思路**：
- 递归参数 `(path, left, right)`；左括号已用 left 个、右括号 right 个，目标 n 对。
- 剪枝：`left < n` 才能加 '('；`right < left` 才能加 ')'。
- `left == n and right == n` 时入答案。
- 复杂度 O(Catalan(n))。

**易错**：剪枝条件 `right < left` 写成 `right <= left`（误判合法）；忘了两个 if 都要尝试（不是 elif）；用 count 计数但没传当前 path。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l2">L2</span> <span class="freq">🔥×5</span> <b>Q29</b> · LC1143 最长公共子序列</summary>

**考察点**：二维 DP；与最长公共子串（连续）的区别。

**关键思路**：
- `dp[i][j]` = text1[:i] 与 text2[:j] 的 LCS 长度。
- 转移：`t1[i-1]==t2[j-1]` 时 `dp[i][j] = dp[i-1][j-1] + 1`；否则 `max(dp[i-1][j], dp[i][j-1])`。
- 边界 `dp[0][*] = dp[*][0] = 0`。
- 空间可滚动数组优化到 O(min(m,n))。

**易错**：与最长公共子串混淆（子串需连续，错位时 dp 清零）；下标 1 偏移；max 漏 dp[i-1][j] 或 dp[i][j-1]。

</details>

<details class="qa qa-handcoding">
<summary><span class="lv lv-l1">L1</span> <span class="freq">🔥×4</span> <b>Q30</b> · LC169 多数元素</summary>

**考察点**：Boyer-Moore 投票算法；为何成立。

**关键思路**：
- 维护 `cand, count`，count=0 时 cand=x、count=1；x == cand 时 count+=1，否则 count-=1。
- 遍历结束 cand 即多数元素（题目保证存在）。
- 复杂度 O(n) / 空间 O(1)。
- 不保证存在时需第二次遍历验证 `count(cand) > n/2`。

**易错**：把题目当"出现 > n/2"反复改成"出现最多"——LC229 是变体（次数 > n/3）；count=0 时不重置 cand 致最终错误。

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
