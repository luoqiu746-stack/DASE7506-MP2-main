# E43/E44 实验交付：算式重算与跨候选证据

本轮最终选择 **repair-full-800/repair-weight0.25**。官方Test **393/467 = 84.1541755889%**，上一版389/467；本轮增加4题。85%至少需397题，目前尚差4题。按预定Train OOF shortlist和Validation规则选定后冻结源码/资产，再做一次最终Test，没有根据该Test结果调参或回选。

**证据有分歧：最终模型Train OOF disagreement比父模型少5题，Validation多1题，Test多4题。** 它不是Train最佳候选，而是预先声明的不同特征路线shortlist中Validation最佳。不能把这次结果描述为每个开发拆分都稳定提升，也不能保证再提高4题。公开Test已随此前冻结版本多次报告，不是全研究过程只看过一次的盲测。

这是实验与复现说明，不是课程报告；课程报告仍另行确认。AI协助实验设计、实现、运行、检查与文档；课程提交应披露并理解实现。

## 结果

| 数据子集 | E42父模型 | E44最终模型 |
|---|---:|---:|
| Train OOF overall | 4385/4755 (92.2187%) | 4380/4755 (92.1136%) |
| Train OOF disagreement | 1744/2114 (82.4976%) | 1739/2114 (82.2611%) |
| Train OOF single_correct | 393/711 (55.2743%) | 386/711 (54.2897%) |
| Train OOF disagreement_single_correct | 366/684 (53.5088%) | 359/684 (52.4854%) |
| Validation overall | 748/798 (93.7343%) | 749/798 (93.8596%) |
| Validation disagreement | 305/355 (85.9155%) | 306/355 (86.1972%) |
| Validation single_correct | 74/117 (63.2479%) | 75/117 (64.1026%) |
| Validation disagreement_single_correct | 71/114 (62.2807%) | 72/114 (63.1579%) |
| Frozen Test | 389/467 (83.2976%) | 393/467 (84.1542%) |

## 新证据与限制

新功能只重算候选已经书写的数值等式，观察重算后是否支持四个现有答案中的另一个。安全AST只支持数值常量、加减乘除和括号，不执行任意文本，不调用外部模型，也不生成新的候选答案。

三个重算策略同时作为特征：local只重算当前式子；protected在后续式子里替换错误中间值，但保护题目已有数字和常见常数；permissive允许更广的数值替换。新路径先去掉行首列表符号，降低把项目符号当负号的误检。116个新增描述符包含修改次数、传播次数、最终数值与各候选答案的一致性、原支持度与修正支持度差异等。

例：候选写`84+21+5=100`，接着`420-100=320`。重算可得到110，并沿用为420-110=310；但100也可能在其他题里代表独立的百分制基数，因此protected策略会保护常见常数，而permissive策略允许传播。这种歧义正是不能硬性替换答案的原因。实际预测仍由训练模型选择现有候选索引。

旧numeric_process_trace仅增加表达式文本返回字段，原计算与旧process特征含义保持。64道Train样本旧/新process特征逐元素完全一致、旧模型预测一致；新特征通过有限值、链式重算、列表符号和候选重排检查。官方Validation父模型预测也保持一致。

Train重算诊断（浮点相对容差1e-5，**不是官方候选选择accuracy**）：

| 策略 | 修改数值的候选数 | 从错误数值修正到参考数值附近 | 从正确数值改坏 | 修正后值出现在现有候选中 |
|---|---:|---:|---:|---:|
| local | 222 | 108 | 17 | 120 |
| protected | 284 | 150 | 18 | 164 |
| permissive | 337 | 160 | 51 | 174 |

相同数字可能表示不同实体，候选也可能选错公式但算术全部正确。重算无法解决全部语义问题，正确最终答案也可以包含错误步骤。因此只将这些统计作为输入证据，不强制优先任何修正答案。提供的Test逐题内容和参考没有用于分析或设计。

## 实验与消融

训练前固定6条路线：原process-meta控制；metadata+旧process+repair深度5种子17/29；同特征深度6；加入embedding的repair-full；metadata+repair但去掉旧process的repair-only。特征维度分别328、444、1468、232。CatBoost800树，300/800快照，lr=.05、leaf L2=5、border64、四线程，E14问题权重；沿用Train问题三折seed7306。另测试两个深度5种子的平均；固定融合权重.25/.5/.75/1，共56配方加父模型。

按Train OOF disagreement/overall/single排序选前三个不同特征家族，再加最佳纯模型和父。Validation同指标选择，完全并列父优先。入围如下：

| 候选 | Train OOF disagreement /2114 | Validation overall /798 | Validation disagreement /355 | Validation single /117 |
| repair2-300/repair-weight0.25 | 1748 | 748 | 305 | 73 |
| repair-full-800/repair-weight0.25 | 1739 | 749 | 306 | 75 |
| repair-only-300/repair-weight0.25 | 1733 | 745 | 302 | 71 |
| repair17-800/repair-weight1 | 1714 | 739 | 296 | 65 |
| frozen-parent-fixed | 1744 | 748 | 305 | 74 |

最终模型Train三折困难题净变化[-1, -3, -1]，配对新增正确18、损失23，描述性bootstrap95%区间[-0.804162724692526, 0.3784295175023652]个百分点。Validation新增正确3、损失2、双方都错47。开发折已被重复使用；bootstrap未校正选择过程，不能当作独立确认。

最终是25%的repair-full-800新树加75%的E42组合，实际成员和尺度以final/config.json为准。不能把整个增益单独归因于重算：该树同时使用嵌入、既有元数据、旧过程特征与新增重算特征，其他消融没有在Validation超过父模型。

## 成本、保护与交付

实际18个CV fit+4个全Train fit；57个OOF配方共享这些模型，不是57次独立训练。成功pipeline 151.012646秒；配对诊断1.643338秒，重算覆盖率诊断1.816297秒。

Test新鲜预测进程1.816524秒+独立评分0.066127秒=**1.882650秒**。本机Apple M5/24GB，四线程CPU；参考Xeon未实测。

final/共9个推理资产，总文件4053732字节；NPZ展开条目+完整原生CBM/config为4050842字节（3.863MiB）；更保守树JSON+NPY统计11666720字节（11.126MiB），均低于16MiB。审计JSON不是推理依赖。

Excel追加255条九列记录（1697–1951行），历史1696行和另一表导出回读保持；新增区域已渲染检查。N/A表示共享成本或未单独计时。源码/模型hash见freeze.json、audit.json、delivery-manifest.json；57个受保护历史模型/官方工具保持不变，单条/批量/问题/候选重排通过。只在main修改原文件，未Commit/Push，无新受跟踪文件，无外部提交。

## 复现冻结推理

```bash
cd /Users/qiu/Documents/courses/7506/DASE7506-MP2-main/code
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
.venv/bin/python predict.py --method submission.py --checkpoint runs/repair-20261027-235139/final --split validation --output runs/e44-reproduced-validation-predictions.json
.venv/bin/python score_predictions.py --split validation --predictions runs/e44-reproduced-validation-predictions.json --output runs/e44-reproduced-validation-score.json
.venv/bin/python predict.py --method submission.py --checkpoint runs/repair-20261027-235139/final --split test --output runs/e44-reproduced-test-predictions.json
.venv/bin/python score_predictions.py --split test --predictions runs/e44-reproduced-test-predictions.json --output runs/e44-reproduced-test-score.json
```

这些命令仅复现已冻结结果，不选择新模型。复制final/全部9个文件；推理不读取训练缓存、ID查表或参考答案。

## 复现本轮训练与选择

ZIP保持code/结构，未重复打包官方数据。解压到包含官方data/的独立项目副本，避免覆盖正在编辑的源文件。包含父E42的Train OOF/折分/模型/结果作为明确的历史依赖；source/是本轮匹配源码，父run/source/保留父版本。

```bash
# 在副本code/目录，Python3.12
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p runs/e44-reproduce
.venv/bin/python -u train_scorer.py --repair --run-dir runs/e44-reproduce --parent-run runs/stability-20261027-233342-retry
.venv/bin/python -u analyze_e22.py --diagnose-repair runs/e44-reproduce
.venv/bin/python -u analyze_e22.py --select-repair runs/e44-reproduce
```

使用尚未启动过的新目录；不要覆盖已有started或Test文件。选择入口在Validation选定并冻结后自动最终Test一次，若父模型保留则校验并继承父结果。机器负载影响时间，CatBoost文件元数据可能导致重训hash变化；匹配的已冻结文件有manifest可复核。requirements.txt和environment-freeze.txt记录依赖。

从Train重新构建父链时，使用如下全新目录，顺序为expanded→kernel→process→stability→repair（各历史选择入口各记录冻结评分，不向训练或选择反馈Test）：

```bash
mkdir -p runs/e32-clean runs/e36-clean runs/e40-clean runs/e42-clean runs/e44-clean
.venv/bin/python -u train_scorer.py --expanded --run-dir runs/e32-clean
.venv/bin/python -u analyze_e22.py --select-expanded runs/e32-clean
.venv/bin/python -u train_scorer.py --kernel --run-dir runs/e36-clean --parent-run runs/e32-clean
.venv/bin/python -u analyze_e22.py --select-kernel runs/e36-clean
.venv/bin/python -u train_scorer.py --process --run-dir runs/e40-clean --parent-run runs/e36-clean
.venv/bin/python -u analyze_e22.py --select-process runs/e40-clean
.venv/bin/python -u train_scorer.py --stability --run-dir runs/e42-clean --parent-run runs/e40-clean
.venv/bin/python -u analyze_e22.py --select-stability runs/e42-clean
.venv/bin/python -u train_scorer.py --repair --run-dir runs/e44-clean --parent-run runs/e42-clean
.venv/bin/python -u analyze_e22.py --select-repair runs/e44-clean
```
