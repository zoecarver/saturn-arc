I built a Python solver with Claude 🤖 in about an hour to test an idea I had—could structured brute-force exploration using ChatGPT-5 as the pattern recognition engine actually tackle ARC-AGI-2 puzzles? The implementation starts with an aggressive brainstorming phase where I prompt GPT to generate twenty different pattern hypotheses, each scored by likelihood from zero to ten. Rather than committing to a single approach, I embrace the combinatorial nature of the problem by generating permutations and combinations of the top five most promising patterns, creating another twenty refined hypotheses. This exhaustive exploration ensures I don't miss potential solutions by being too narrow in initial thinking. The solver then systematically tests these patterns through explicit verification phases where GPT must apply each candidate pattern to training examples and signal success with "PATTERN FOUND" or failure with "PATTERN FAILED."

After testing the top patterns from initial rounds, the system generates three new patterns based on insights gained from failures, then tests those as well. This iterative refinement process combines brute-force exploration with adaptive learning, working across multiple levels of abstraction from low-level pixel patterns to high-level transformation rules. When a pattern is marked as found, I perform final validation where GPT applies the discovered pattern to generate output for the test input, which I then compare against actual test output. This two-stage verification ensures patterns that work on training data actually generalize to the test case. Maintaining conversational context throughout ensures GPT can build on previous attempts and learn from what didn't work. The approach essentially implements a poor man's version of what François Chollet describes as guided search through the vast space of possible pattern transformations, using deep learning's pattern recognition to suggest promising candidates while systematic search assembles these building blocks into concrete solutions.

I achieved a 72% score on 60 random samples from ARC-AGI-2 open source problems. This is likely illegitimate, and there's good reason for skepticism. The core issue is data contamination: I tested on 60 problems from the "open source dataset," which ChatGPT-5 almost certainly encountered during training. The model likely learned ARC patterns, transformation rules, and problem structures, even if not exact solutions. The solutions may be encoded deep in ChatGPT-5's weights but require extensive prompting to extract. ChatGPT fails most problems on first attempts, suggesting solutions aren't readily accessible, but my method's ~42 pattern exploration acts as a sophisticated retrieval algorithm, systematically activating different neural pathways until the right combination surfaces the buried knowledge. The fact that extensive exploration is needed suggests the model is doing sophisticated pattern matching on familiar data rather than demonstrating genuine fluid intelligence or novel reasoning. Testing on guaranteed unseen data is needed for legitimate validation. Each problem cost roughly $0.30 in tokens to solve. Even if the 72% score reflects sophisticated extraction of training data rather than genuine breakthrough reasoning, the methodology itself demonstrates that systematic exploration can dramatically improve AI performance and we may be underestimating latent AI capabilities due to poor retrieval methods.


### Setup and running

1. Clone repo
2. `export OPENAI_API_KEY=...`
3. `python3 arc_solver.py ARC-AGI-2/data/training/00576224.json`
4. Or `python3 run_batch.py 10` (runs 10 random problems)


### Results 

```
================================================================================
BATCH RESULTS SUMMARY
================================================================================
Total tasks: 60
Successful: 43 (71.7%)
Failed: 17 (28.3%)

Detailed Results:
Task                 Result     Time (s)  
----------------------------------------
782b5218             ❌ FAIL     971.97    
3befdf3e             ❌ FAIL     1028.88   
6b9890af             ⚠️ ERROR    338.54    
8a6d367c             ✅ PASS     532.59    
6855a6e4             ❌ FAIL     1486.63   
a57f2f04             ✅ PASS     1384.07   
ce4f8723             ✅ PASS     298.88    
3c9b0459             ✅ PASS     313.14    
84ba50d3             ❌ FAIL     1468.58   
f76d97a5             ✅ PASS     489.36    
5623160b             ✅ PASS     1122.98   
e2092e0c             ✅ PASS     917.31    
b60334d2             ✅ PASS     375.91    
8e1813be             ❌ FAIL     756.23    
b0f4d537             ❌ FAIL     1265.08   
4347f46a             ✅ PASS     435.09    
77fdfe62             ✅ PASS     512.10    
484b58aa             ❌ FAIL     706.11    
a68b268e             ✅ PASS     489.17    
6d0160f0             ❌ FAIL     1307.81   
bd283c4a             ✅ PASS     879.41    
cdecee7f             ✅ PASS     489.26    
e50d258f             ✅ PASS     403.39    
6ffe8f07             ✅ PASS     1098.35   
963e52fc             ✅ PASS     272.59    
88a10436             ✅ PASS     501.36    
b2862040             ❌ FAIL     1152.55   
494ef9d7             ✅ PASS     1100.92   
1e0a9b12             ✅ PASS     351.23    
1bfc4729             ✅ PASS     520.60    
2c608aff             ✅ PASS     1035.19   
df978a02             ❌ FAIL     1162.49   
6e19193c             ❌ FAIL     1149.27   
140c817e             ✅ PASS     760.77    
0e671a1a             ✅ PASS     789.19    
d687bc17             ✅ PASS     998.32    
855e0971             ✅ PASS     725.30    
833966f4             ✅ PASS     411.55    
97239e3d             ✅ PASS     1479.74   
a740d043             ✅ PASS     244.83    
aaef0977             ✅ PASS     564.82    
48d8fb45             ✅ PASS     593.51    
f3cdc58f             ✅ PASS     464.69    
af726779             ❌ FAIL     967.98    
54db823b             ✅ PASS     468.69    
6cf79266             ✅ PASS     633.65    
c909285e             ❌ FAIL     648.33    
92e50de0             ✅ PASS     689.95    
cf98881b             ✅ PASS     499.24    
a2d730bd             ❌ FAIL     988.74    
e73095fd             ✅ PASS     1012.40   
825aa9e9             ❌ FAIL     1127.60   
25d487eb             ✅ PASS     534.12    
36d67576             ❌ FAIL     847.36    
d4f3cd78             ✅ PASS     413.90    
e734a0e8             ✅ PASS     848.22    
8f2ea7aa             ✅ PASS     932.54    
0c9aba6e             ✅ PASS     625.34    
4938f0c2             ✅ PASS     570.39    
623ea044             ✅ PASS     684.85    
```
