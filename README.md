I built a Python solver with Claude 🤖 in about an hour to test an idea I had—could structured brute-force exploration using ChatGPT-5 and Claude 4 (Sonnet) as the pattern recognition engine actually tackle ARC-AGI-2 puzzles? The implementation starts with an aggressive brainstorming phase where I prompt GPT to generate twenty different pattern hypotheses, each scored by likelihood from zero to ten. Rather than committing to a single approach, I embrace the combinatorial nature of the problem by generating permutations and combinations of the top five most promising patterns, creating another twenty refined hypotheses. This exhaustive exploration ensures I don't miss potential solutions by being too narrow in initial thinking. The solver then systematically tests these patterns through explicit verification phases where GPT must apply each candidate pattern to training examples and signal success with "PATTERN FOUND" or failure with "PATTERN FAILED."

After testing the top patterns from initial rounds, the system generates three new patterns based on insights gained from failures, then tests those as well. This iterative refinement process combines brute-force exploration with adaptive learning, working across multiple levels of abstraction from low-level pixel patterns to high-level transformation rules. When a pattern is marked as found, I perform final validation where GPT applies the discovered pattern to generate output for the test input, which I then compare against actual test output. This two-stage verification ensures patterns that work on training data actually generalize to the test case. Maintaining conversational context throughout ensures GPT can build on previous attempts and learn from what didn't work. The approach essentially implements a poor man's version of what François Chollet describes as guided search through the vast space of possible pattern transformations, using deep learning's pattern recognition to suggest promising candidates while systematic search assembles these building blocks into concrete solutions.

I achieved a 72% score on 60 random samples from ARC-AGI-2 open source problems. This is likely illegitimate, and there's good reason for skepticism. The core issue is data contamination: I tested on 60 problems from the "open source dataset," which both models almost certainly encountered during training. The model likely learned ARC patterns, transformation rules, and problem structures, even if not exact solutions. The solutions may be encoded deep in the model's weights but require extensive prompting to extract. Both models fail most problems on first attempts, suggesting solutions aren't readily accessible, but my method's ~42 pattern exploration acts as a sophisticated retrieval algorithm, systematically activating different neural pathways until the right combination surfaces the buried knowledge. The fact that extensive exploration is needed suggests the model is doing sophisticated pattern matching on familiar data rather than demonstrating genuine fluid intelligence or novel reasoning. Testing on guaranteed unseen data is needed for legitimate validation. Each problem cost roughly $0.30 in tokens to solve. Even if the 72% score reflects sophisticated extraction of training data rather than genuine breakthrough reasoning, the methodology itself demonstrates that systematic exploration can dramatically improve AI performance and we may be underestimating latent AI capabilities due to poor retrieval methods.


### Setup and running

1. Clone repo
2. `export OPENAI_API_KEY=...`
3. `python3 arc_solver.py ARC-AGI-2/data/training/00576224.json`
4. Or `python3 run_batch.py 10` (runs 10 random problems)
4. Use help menu to see other options (what dataset to run, run in parallel, etc.)


### Results 

From the `training` dataset...

```
================================================================================
BATCH RESULTS SUMMARY (train)
================================================================================
Total tasks: 20
Successful: 10 (50.0%)
Failed: 10 (50.0%)
Total time: 11143.67s
Total prompts sent: 109

Detailed Results:
Task                 Result     Time (s)   Prompts   
--------------------------------------------------
332efdb3             ✅ PASS     303.70     3         
91714a58             ✅ PASS     423.16     4         
85fa5666             ❌ FAIL     672.00     8         
e7a25a18             ❌ FAIL     720.66     8         
c9e6f938             ✅ PASS     240.50     3         
4acc7107             ❌ FAIL     1017.93    8         
15113be4             ❌ FAIL     706.69     8         
93b4f4b3             ❌ FAIL     920.90     8         
6855a6e4             ⚠️ ERROR    6.76       0         
bda2d7a6             ✅ PASS     348.29     3         
b15fca0b             ❌ FAIL     662.96     8         
72a961c9             ✅ PASS     458.53     3         
5587a8d0             ✅ PASS     255.04     3         
4938f0c2             ✅ PASS     396.59     3         
18419cfa             ❌ FAIL     684.13     8         
db118e2a             ✅ PASS     553.23     4         
7039b2d7             ✅ PASS     559.00     8         
23b5c85d             ✅ PASS     343.03     3         
6aa20dc0             ❌ FAIL     1006.83    8         
c9680e90             ❌ FAIL     863.73     8       

================================================================================
BATCH RESULTS SUMMARY (eval)
================================================================================
Total tasks: 20
Successful: 1 (5.0%)
Failed: 19 (95.0%)
Total time: 12000.97s
Total prompts sent: 155

Detailed Results:
Task                 Result     Time (s)   Prompts   
--------------------------------------------------
3a25b0d8             ❌ FAIL     356.19     8         
2d0172a1             ❌ FAIL     404.27     8         
7b80bb43             ❌ FAIL     473.04     8         
8f3a5a89             ❌ FAIL     542.00     8         
5545f144             ❌ FAIL     682.09     8         
b9e38dc0             ❌ FAIL     551.46     8         
e376de54             ❌ FAIL     580.71     8         
f560132c             ❌ FAIL     737.23     8         
6e453dd6             ❌ FAIL     741.18     8         
a47bf94d             ❌ FAIL     755.82     8         
d8e07eb2             ❌ FAIL     506.34     8         
4e34c42c             ❌ FAIL     637.78     8         
981571dc             ❌ FAIL     424.04     8         
269e22fb             ❌ FAIL     585.69     8         
b0039139             ❌ FAIL     570.31     8         
45a5af55             ✅ PASS     562.09     3         
65b59efc             ❌ FAIL     486.14     8         
faa9f03d             ❌ FAIL     425.13     8         
135a2760             ❌ FAIL     1327.54    8         
1818057f             ❌ FAIL     651.92     8        

```
