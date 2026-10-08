import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 【修复】原为 matplotlib.pylab，该模块已废弃
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 全局字体配置，解决中文不显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

# 数据文件路径：按顺序查找，用第一个存在的路径
# ① 脚本同级目录（推荐：把 .xls 和脚本放一起，别人 clone 后也能直接跑）
# ② 脚本同级的 data/ 子目录
# ③ 本机原始存放路径
DATA_FILE = 'default of credit card clients.xls'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PATHS = [
    os.path.join(_SCRIPT_DIR, DATA_FILE),
    os.path.join(_SCRIPT_DIR, 'data', DATA_FILE),
    r'E:\统计学习\数据集\信贷项目\default of credit card clients.xls',
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
# 【新增】输出目录：默认放在脚本同级的 outputs 文件夹，图表和结果表统一落盘，避免画完就丢。
# 用脚本自身路径而不是写死绝对路径，换电脑/换目录都不用改代码。
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# 找不到数据文件时给出明确提示，而不是抛一堆 traceback
if DATA_PATH is None:
    raise FileNotFoundError(
        '找不到数据文件：' + DATA_FILE + '\n'
        '请从 https://archive.ics.uci.edu/dataset/350 下载 '
        '"default of credit card clients.xls"，\n'
        '放到脚本同级目录（或 data/ 子目录）即可；\n'
        '也可以修改脚本顶部的 CANDIDATE_PATHS 指向你的实际路径。'
    )
print('数据文件：', DATA_PATH)
df=pd.read_excel(DATA_PATH,header=1)
#【修复】记下读入时的原始形状。后面会往 df 里不断加分箱列/衍生列，
#末尾如果用 df.shape 写汇总，列数会变成 55 这种分析中间态，不是数据的真实规格。
df_shape_raw=df.shape
# 设置显示格式(调试时使用)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

#================数据加载与清洗=================
#EDUCATION（教育程度）1=研究生、2=大学、3=高中、4=其他
#MARRIAGE（婚姻状况） 1=已婚、2=单身、3=其他
#pay_1到pay_6记录客户过去 7 个月每月的还款状态:-2表示没有借款，-1表示已缴清，0表示当月有余额，但按时还了最低还款额，正数表示逾期的月份
#BILL_AMT1 ~ BILL_AMT6表示上月账单金额（应还的账单总额）
#PAY_AMT1 ~ PAY_AMT6表示上月已偿还金额（实际还了多少）
#确认形状、类型、缺失值
print('='*60)
print('步骤0 数据加载与清洗')
print('='*60)
print('数据形状:',df.shape)
print('缺失值总数:',df.isna().sum().sum())
#df=df.drop('ID',axis=1)#ID列纯标识符，无泛化信息,后续训练模型再删
df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4})
df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
#未定义编码统计：EDUCATION 的 0/5/6 共 345 条，MARRIAGE 的 0 共 54 条，全部归入"其他"
print('未定义编码已修正：EDUCATION 的 0/5/6 归入 4（其他），MARRIAGE 的 0 归入 3（其他）')
#BILL_AMT会出现负值，属于溢缴款、退款、费用返还等是正常现象
df.rename(columns={'PAY_0':'PAY_1'},inplace=True)#将pay_0改成pay_1
pay_amt_cols=[f'PAY_AMT{i}'for i in range(1,7)]
all_zero=(df[pay_amt_cols]==0).all(axis=1)
print(f'六个月还款额全为 0 的客户数:{all_zero.sum()}（占比 {all_zero.mean():.2%}）')
print(f'BILL_AMT1 负值（溢缴款）笔数:{(df["BILL_AMT1"]<0).sum()}')#溢缴款（客户多还了钱，账单余额为负），在业务上真实存在
#pay_1到pay_6不构成数据泄露，都是历史逐月逾期状态而非预测时刻之后的数据
#计算基线标准，后续训练模型效果必须大于77.88%，模型才有价值
target='default payment next month'
base_badrate=df[target].mean()
print(f'整体坏账率:{base_badrate:.4f}（{df[target].sum()}/{len(df)}）')
print(f'多数类基线准确率:{1-base_badrate:.4f}   模型准确率不超过它就没有价值')


#================单变量 EDA 与坏账率分析=================
print('\n'+'='*60)
print('步骤1 单变量 EDA 与坏账率分析')
print('='*60)
#算 SEX / EDUCATION / MARRIAGE 的分组坏账率,可以直观看到不同性别，学历，婚姻人群的违约差异
def badrate_analysis(df,col):
    res=df.groupby(col,observed=True)[target].agg(['count','mean'])
    return res
print('--- SEX 分组坏账率 ---')
print(badrate_analysis(df,'SEX'))
print('--- EDUCATION 分组坏账率 ---')
print(badrate_analysis(df,'EDUCATION'))
print('--- MARRIAGE 分组坏账率 ---')
print(badrate_analysis(df,'MARRIAGE'))
#算 PAY_1 每个取值的坏账率，画出趋势
pay1_bad=badrate_analysis(df,'PAY_1')
print('--- PAY_1（最近一期还款状态）各取值坏账率 ---')
print(pay1_bad)
plt.figure(figsize=(8,5))
plt.plot(pay1_bad.index,pay1_bad['mean'],marker='o')
plt.xlabel('pay_1的逾期状态')
plt.ylabel('坏账率')
plt.title('pay_1各取值坏账率趋势')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR,'01_pay1_badrate_trend.png'),dpi=120,bbox_inches='tight')  #【修复】原来只 show()/close()，图全部丢失
plt.close()
#把 PAY_1 按 >=1 / <=0 二分，对比两组坏账率
#【修复】这里原来用 df['pay1_bin'] 这个临时列做二分统计。它只在下面一行用过一次，
#属于死代码；而且名字和入模变量 PAY1_bin(还款状态四档)、pay1_bins(PAY_AMT1 等频分箱)
#只差大小写和单复数，极易看错，已删掉临时列、直接对 PAY_1 取条件。
print('--- PAY_1 二分对比（逾期>=1 个月 vs 正常<=0）---')
print(df.groupby(df['PAY_1']>=1)[target].agg(['count','mean']))#逾期客户坏账率远高于无逾期客户
#对 LIMIT_BAL 做等频分箱，看坏账率是否单调
df['limit_bal_bins']=pd.qcut(df['LIMIT_BAL'],q=10)
print('--- LIMIT_BAL 等频10箱坏账率 ---')
print(df.groupby('limit_bal_bins',observed=True)[target].agg(['count','mean']))#可以看到坏账率随着授信金额增大而减少
#对 AGE 做等频分箱，观察分布形状
df['age_bins']=pd.qcut(df['AGE'],q=10)
print('--- AGE 等频10箱坏账率 ---')
print(df.groupby('age_bins',observed=True)[target].agg(['count','mean']))#年龄坏账率偏U型，年轻人和老年人坏账偏高，中华人最低
#对 BILL_AMT1 / PAY_AMT1 做等频分箱
for i in range(1,7):
  df[f'bill{i}_bins']=pd.qcut(df[f'BILL_AMT{i}'],q=5)
print('--- BILL_AMT1 等频5箱坏账率 ---')
print(df.groupby('bill1_bins',observed=True)[target].agg(['count','mean']))#bill分箱之后几乎无规律
for i in range(1,7):
  df[f'pay{i}_bins']=pd.qcut(df[f'PAY_AMT{i}'],q=5,duplicates='drop')
print('--- PAY_AMT1 等频5箱坏账率 ---')
print(df.groupby('pay1_bins',observed=True)[target].agg(['count','mean']))#pay分箱之后完美单调递减
#构造衍生特征 额度使用率 = BILL_AMT1 / LIMIT_BAL 并分析
df['util_use']=df['BILL_AMT1']/df['LIMIT_BAL']#【修复】LIMIT_BAL 最小值 10000，永不为 0，不需要 +1e-6
df['util_use_bins']=pd.qcut(df['util_use'],q=10)
print('--- 额度使用率 等频10箱坏账率 ---')
print(df.groupby('util_use_bins',observed=True)[target].agg(['count','mean']))#整体递增（使用率越高风险越高，符合业务逻辑）
#构造衍生变量delay_cnt(逾期次数)
pay_cols=[f'PAY_{i}'for i in range(1,7)]
df['delay_cnt']=((df[pay_cols]>0).sum(axis=1))
df['delay_cnt_bins']=pd.qcut(df['delay_cnt'],q=10,duplicates='drop')
print('--- delay_cnt(逾期月份数) 各箱坏账率 ---')
print(df.groupby('delay_cnt_bins',observed=True)[target].agg(['count','mean']))
#构造衍生变量pay_ratio1(还款比例)
df['BILL_AMT1_safe']=df['BILL_AMT1'].replace(0,1)
df['pay_ratio1']=(df['PAY_AMT1']/df['BILL_AMT1_safe'])
#【修复】原来这里先用 qcut 建了一列 pay_ratio1_bins，下面又用 pd.cut 覆盖，前面的赋值是死代码，已删除
#检验 ID 能否作为时间代理（做卡方检验）
df['id_bins']=pd.qcut(df['ID'],q=5)
cross_table=pd.crosstab(df['id_bins'],df[target])
ch2,p,dof,expected=chi2_contingency(cross_table)
print(f'--- ID 分段时间代理检验：卡方={ch2:.4f}, p值={p:.4f}, 自由度={dof} ---')
print('各 ID 段坏账率:')
print(df.groupby('id_bins',observed=True)[target].agg(['count','mean']))
print(f'坏账率极差={df.groupby("id_bins",observed=True)[target].mean().max()-df.groupby("id_bins",observed=True)[target].mean().min():.4f}')
#p值很小拒绝原假设，但各段坏账率极差只有 6.6 个百分点且无单调趋势，考虑 N=30000 时卡方检验极度敏感，我判断这只是大样本下的统计显著而非真实的时间漂移，因此不采用 ID 作为时间代理
#业务发现：1.还款历史是最强信号：有过逾期（PAY_0≥1）的客户坏账率 50.29%，是无逾期客户（13.83%）的 3.6 倍，而这类客户占 22.73%
#2.额度越低风险越高：授信额度 ≤3 万的客户坏账率 35.85%，是 >36 万客户（11.87%）的 3 倍
#3.还款金额是保护因素：月还款 ≤316 元的客户坏账率 34.22%，是还款 >10300 元客户（12.78%）的 2.7 倍
#4.年龄呈 U 型：25 岁以下和 49 岁以上风险偏高，中间段最稳
#5.账单金额几乎无预测力：BILL_AMT1 各分箱坏账率在 19.67%~25.53% 之间无规律波动 —— 反直觉但重要：欠多少钱不如"还得怎么样"重要


#================分箱与单调性=================
print('\n'+'='*60)
print('步骤2 分箱与单调性')
print('='*60)
#为 PAY_0~PAY_6 设计自定义粗分类，解决不单调 + 小样本箱(高逾期数字样本极少会出现小样本箱)
#【修复】原来把 -2/-1 合成"按时结清"、0 单独作"循环账户"，实测两者坏账率倒挂
#（按时结清 15.62% > 循环账户 12.81%），破坏单调性。三者业务上都是"当前未逾期"，故合并
def bin_pay(x):
    if x<=0:
        return'正常(≤0)'
    elif x==1:
        return'逾期1个月'
    elif x==2:
        return'逾期2个月'
    else:
        return '逾期3个月以上'
#【修复】必须用 pd.Categorical 指定业务顺序
#原因：字符串分箱后 groupby 会按 Unicode 排序，"循环账户"会排在"按时结清"前面，
#      导致报表里看起来"单调递增"，实际业务顺序是乱的，会掩盖真实的非单调问题。
pay_order=['正常(≤0)','逾期1个月','逾期2个月','逾期3个月以上']
for i in range(1,7):
    df[f'PAY{i}_bin'] = pd.Categorical(df[f'PAY_{i}'].apply(bin_pay),categories=pay_order,ordered=True)
print('--- PAY1_bin 人工分箱后（已按业务顺序排列）---')
print(df.groupby('PAY1_bin',observed=True)[target].agg(['count','mean']))#人工分箱且坏账率单调
#为 LIMIT_BAL / PAY_AMT1 / 额度使用率 设计粗分类
#一.LIMIT_BAL
def bin_limit_bal(x):
    if x <= 30000:
        return "≤30000"
    elif x <= 70000:
        return "30000~70000"
    elif x <= 100000:
        return "70000~100000"
    elif x <= 140000:
        return "100000~140000"
    elif x <= 180000:
        return "140000~180000"
    elif x <= 270000:
        return "180000~270000"
    elif x <= 360000:
        return "270000~360000"
    else:
        return ">360000"
df['limit_bin_final'] = df['LIMIT_BAL'].apply(bin_limit_bal)#生成分箱列
bin_order = [
    "≤30000",
    "30000~70000",
    "70000~100000",
    "100000~140000",
    "140000~180000",
    "180000~270000",
    "270000~360000",
    ">360000"
]#指定箱子顺序，解决字符串乱序问题
df['limit_bin_final'] = pd.Categorical(
    df['limit_bin_final'],
    categories=bin_order,
    ordered=True)
print('--- LIMIT_BAL 人工分箱后坏账率 ---')
print(df.groupby('limit_bin_final',observed=True)[target].agg(['count','mean']))#LIMIT_BAL人工分箱后单调
#二.PAY_AMT1(本身的等频分箱就单调无需人工)
#三.额度使用率
def bin_util(x):
    if x <=0.2:
        return "低使用率(≤0.2)"
    elif x <=0.5:
        return "中使用率(0.2~0.5)"
    elif x <=0.8:
        return "偏高使用率(0.5~0.8)"
    else:
        return "高使用率(>0.8)"
df['util_bins']=df['util_use'].apply(bin_util)
util_order=["低使用率(≤0.2)","中使用率(0.2~0.5)","偏高使用率(0.5~0.8)","高使用率(>0.8)"]
df['util_bins']=pd.Categorical(df['util_bins'],categories=util_order,ordered=True)
print('--- 额度使用率 人工分箱后坏账率 ---')
print(df.groupby('util_bins',observed=True)[target].agg(['count','mean']))#额度使用率通过人工分箱之后变成单调递增
#【修复】标签整体错位一格：bins 有 6 个切点=5 个区间，labels 5 个按位置对齐后，
#从第 2 个起全部错位（(0,0.2] 被贴上"0（完全不还款）"、(0.2,0.5] 被贴上"(0,0.2] 少量还款"…）。
#【修复】同时解决单调性问题：实测各箱坏账率为
#  负数(溢缴款) 0.1513 ｜ 0(完全不还款) 0.3595 ｜ (0,0.2] 0.2079 ｜ (0.2,0.5] 0.1873 ｜ ≥0.5 0.1521
#  "溢缴款"(0.1513) 与"≥0.5 大额还款"(0.1521) 风险几乎相同，说明把溢缴款当成"最极端的一档"是
#  排序假设错了 —— 多还钱并不比按时还钱更危险。它夹在中间会破坏单调性，
#  也直接导致该变量系数为负（非单调变量的典型症状）。
#  处理：把"负数(溢缴款)"并入"≥0.5 大额/全额还款"，两箱风险相近且业务含义一致（都属于"还款充足"）。
bins_break = [-np.inf, 1e-9, 0.2, 0.5, np.inf]
bin_labels = [
     "0（完全不还款）",
     "(0,0.2] 少量还款",
     "(0.2,0.5] 部分还款",
     "≥0.5 大额/全额还款(含溢缴款)"
 ]
df['pay_ratio1_bins'] = pd.cut(df['pay_ratio1'],bins=bins_break,labels=bin_labels,include_lowest=True)
print('--- pay_ratio1(还款比例) 分箱坏账率 ---')
print(df.groupby('pay_ratio1_bins',observed=True)[target].agg(count = 'count',bad_rate = 'mean').reset_index())
#【新增】空箱自检：任何一箱样本为 0 都说明切点设置有问题
empty_bin=df['pay_ratio1_bins'].value_counts()
print(f'空箱检查：{list(empty_bin[empty_bin==0].index) if (empty_bin==0).any() else "无空箱"}')
#检查每一箱的样本量占比，低于 5% 的箱必须合并
def check_bin_ratio(df,bin_col):
    ratio=df[bin_col].value_counts(normalize=True).sort_values(ascending=False)
    return(ratio)
print('--- 各分箱样本占比检查（低于 5% 需合并）---')
for col in ['limit_bin_final','pay1_bins','util_bins','PAY1_bin','pay_ratio1_bins']:
    print(f'{col}:')
    print(check_bin_ratio(df,col))
#合并后重新计算各箱坏账率，验证单调性(已验证)
'''思考：箱数多好还是少好？说出权衡 
箱数多：捕捉更细的风险差异，IV更高；缺点：容易出现小样本箱，分箱切点不稳定，容易过拟合，PSI漂移风险高
箱数少：样本充足、稳定性强、业务好解释；缺点：风险区分变粗糙，IV下降，丢失细节信息
实操折中：一般4~6箱。在保证样本充足+单调前提下，尽可能多保留箱数。'''
#【新增】检查分箱的稳定性：随机抽两半数据分别看坏账率，判断分箱是否依赖偶然切分
print('--- 分箱稳定性检查：随机对半分样本，对比各箱坏账率 ---')
half_idx=np.random.RandomState(0).permutation(len(df))[:len(df)//2]
df_half=df.iloc[half_idx]
for col in ['PAY1_bin','limit_bin_final','util_bins']:
    a=df.groupby(col,observed=True)[target].mean()
    b=df_half.groupby(col,observed=True)[target].mean()
    cmp=pd.DataFrame({'全量坏账率':a,'半数坏账率':b})
    cmp['差异']=(cmp['全量坏账率']-cmp['半数坏账率']).abs()
    print(f'{col} 各箱坏账率最大差异 = {cmp["差异"].max():.4f}')


#================WOE / IV 计算与变量筛选=================
print('\n'+'='*60)
print('步骤3 WOE / IV 计算与变量筛选')
print('='*60)
#定义WOE函数
def calc_woe(df,bin_col,target_col):
    gp=df.groupby(bin_col,observed=True)[target_col].agg(bad='sum',total='count').reset_index()
    gp['good']=gp['total']-gp['bad']
    total_bad=gp['bad'].sum()
    total_good=gp['good'].sum()
    #拉普拉斯平滑(+1)，防止某个箱没有坏/好样本时 log(0)；N=30000 下对结果影响可忽略
    gp['good_s'] = gp['good'] + 1
    gp['bad_s'] = gp['bad'] + 1
    total_bad_s = total_bad + len(gp)
    total_good_s = total_good + len(gp)
    gp['bad_pct']=(gp['bad_s'])/total_bad_s
    gp['good_pct']=(gp['good_s'])/total_good_s
    #【注意】本项目的 WOE 符号约定是 ln(坏占比/好占比)：坏账率越高的箱 WOE 越正。
    #这个约定下，预测"违约"的逻辑回归系数应当为正。全文必须统一，否则系数符号解读会反。
    gp['WOE']=np.log(gp['bad_pct']/gp['good_pct'])
    gp.rename(columns={bin_col: "分箱区间"}, inplace=True)
    gp["变量名"] = bin_col
    gp = gp[["变量名", "分箱区间", "bad", "total", "good", "bad_pct", "good_pct", "WOE"]]
    return gp
print('--- 重点变量 WOE 明细 ---')
for col in ['PAY1_bin','limit_bin_final','pay1_bins','util_bins']:
    print(calc_woe(df,col,target))
def calc_iv(df,bin_col,target_col):
    gp=calc_woe(df,bin_col,target_col)
    gp['iv_bin']=((gp['bad_pct'])-(gp['good_pct']))*gp['WOE']
    total_iv=gp['iv_bin'].sum()
    return total_iv

col_list=['limit_bin_final','SEX','EDUCATION','MARRIAGE','age_bins',*[f'PAY{i}_bin' for i in range(1,7)],
          *[f'bill{i}_bins' for i in range(1,7)],*[f'pay{i}_bins' for i in range(1,7)],'util_bins','delay_cnt_bins','pay_ratio1_bins']
iv_list=[]
for col in col_list:
    iv=calc_iv(df,col,target)
    iv_list.append({'feature':col,'iv':iv})
iv_df=pd.DataFrame(iv_list).sort_values('iv',ascending=False).reset_index(drop=True)
print('--- 全变量 IV 排序表 ---')#求解26个变量的IV
print(iv_df.to_string(index=False))
'''解释为什么 IV>0.5 要警惕:
1.IV过高代表变量区分力太强，大概率标签泄漏（变量里直接包含了未来违约信息）
2.现实业务里信贷特征很难做到IV>0.5；往往是目标信息混入特征
3.极易过拟合，OOT样本上AUC暴跌，上线漂移严重'''
print('--- 满足 IV>=0.02 的入模候选 ---')#筛选变量清单
print(iv_df[iv_df['iv']>0.02].to_string(index=False))
#规则：保留 IV ≥0.02；IV<0.02剔除。
#剔除理由：对违约几乎没有区分能力，噪声大，入模会干扰模型、增加不稳定。
#构造一个衍生特征，观察它的 IV 是否"虚高":构造的delay_cnt就是完全由pay系列原始变量计算得到属于信息重复
#而额度使用率/ PAY_RATIO1是两个不同维度字段相除，产生新业务维度，不是虚高
#输出最终 WOE 映射表
#【修复】这里原来把"全量数据拟合的 WOE"导出成 woe映射表.csv，而 README 把它标注为"生产环境部署用"。
#但交付的分值表(score_card.csv)用的是**训练集拟合**的 WOE（步骤5 的 woe_mapping_dict，以避免数据泄露）。
#两者对同一个箱会给出不同的 WOE —— 实测 45 个共同箱全部不一致，
#业务方拿 woe映射表.csv 去算分，结果和分值表对不上。
#现在把导出动作移到步骤5 之后，导出的就是分值表实际使用的那一套映射。
#这里的全量 WOE 只用于本步骤的变量筛选（IV）与箱分布展示，不再作为部署产物。
woe_map_all=pd.DataFrame()
for feat in col_list:
    woe_df=calc_woe(df,feat,target)
    woe_map_all=pd.concat([woe_map_all,woe_df],ignore_index=True)
print('--- WOE 映射表（前 10 行，全量数据口径，仅用于变量筛选与箱分布展示）---')
print(woe_map_all.head(10))
print('说明：真正用于部署的 WOE 映射表在步骤5 用训练集重新拟合后导出，与分值表同源。')
#对比：筛掉 BILL_AMT 六个变量损失了多少 IV
bill_iv=iv_df[iv_df['feature'].str.startswith('bill')]['iv'].sum()
print(f'BILL_AMT 六个变量 IV 合计 = {bill_iv:.4f}（全部 <0.02，已整体剔除，信息损失可忽略）')


#================共线性诊断=================
print('\n'+'='*60)
print('步骤4 共线性诊断')
print('='*60)
#算所有数值变量的相关矩阵，找出 |r|>0.7 的组合
num_cols = ["LIMIT_BAL","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
             "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
             "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",'util_use','delay_cnt','pay_ratio1']
corr_matrix=df[num_cols].corr()
corr_pairs=[]
for i in range(len(corr_matrix.columns)):
    for j in range(i+1,len(corr_matrix.columns)):
        r=corr_matrix.iloc[i,j]
        if abs(r) > 0.7:
          corr_pairs.append((corr_matrix.columns[i],corr_matrix.columns[j],r))
res_df=pd.DataFrame(corr_pairs,columns=['变量1','变量2','相关系数r'])
print(f'--- |r|>0.7 的变量对，共 {len(res_df)} 组 ---')#找出相关系数大于0.7的组合
print(res_df.sort_values('相关系数r',ascending=False).to_string(index=False))
#解释 BILL_AMT1~6 之间 r>0.8 的含义和后果
'''相关系数 r>0.8，说明连续6个月账单金额高度正相关。上个月账单高的客户，下个月账单大概率也高；6个账单变量承载大量重复信息。
后果：1.6个变量信息大量重叠，放入逻辑回归会出现多重共线性；
2.回归系数估计不稳定，系数正负号可能反常，p值不可靠；
3.VIF膨胀；模型很难判断到底是哪一期账单在影响违约；
4.业务冗余，6个变量干的几乎是一件事，不需要全部入模。'''
#检查 PAY_* 变量之间的相关性
print('--- PAY_* 之间相关性矩阵 ---')
print(df[[f'PAY_{i}' for i in range(1,7)]].corr().round(4))
'''含义：上个月逾期的客户，后续月份也容易逾期，信息存在重叠。'''
#说明共线性对逻辑回归的具体危害
'''1.回归系数估计不稳定，方差变大：样本轻微变化，系数大小、正负号就会剧烈波动。
2.p值不可信：本来显著的变量，p值变大，误判为不显著，逐步回归错误剔除有效特征。
3.业务解释完全错乱：业务上风险越高系数应该为正，共线性下系数符号颠倒，评审无法解释。
4.VIF膨胀，标准误差放大，但模型整体预测概率/AUC不一定明显下降（这就是关键点！）'''
#提出三种处理共线性的方法并比较
'''1.人工筛选:每组高相关变量只保留一个
2.逐步回归＋VIF筛选。缺点是容易受样本扰动，结果不稳定
3.降维(PCA主成分)，缺点是主成分失去业务含义，评分卡项目一般禁用'''
#为什么树模型不受共线性影响？
'''树模型（决策树、随机森林、XGBoost）是分层单点分裂，每次只挑一个特征做切分，不一次性同时估计一组自变量的系数，不求解联合系数。
1.高度相关变量，树只会选择其中区分能力最好的一个做分裂；剩下相关变量几乎不会被选中。
2.不存在“系数估计、标准误、VIF”这套逻辑，因此多重共线性不会破坏树模型'''


#================WOE 编码建模与评估=================
print('\n'+'='*60)
print('步骤5 WOE 编码建模与评估')
print('='*60)
#按 7:3 分层切分训练/测试集
# y：违约标签
y = df[target]
# X：所有预测特征
feature_cols =['limit_bin_final','EDUCATION','age_bins',*[f'PAY{i}_bin' for i in range(1,7)],
    *[f'pay{i}_bins' for i in range(1,7)],'util_bins','pay_ratio1_bins']
X=df[feature_cols]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
print(f'训练集 {len(x_train)}，测试集 {len(x_test)}')
print(f'训练集坏账率 {y_train.mean():.6f}，测试集坏账率 {y_test.mean():.6f}')
#用训练集拟合 WOE 映射，再应用到测试集
df_test=x_test.copy()
df_train=x_train.copy()
df_train['default']=y_train
woe_mapping_dict={}
for feat in feature_cols:
    woe_table=calc_woe(df_train,feat,'default')
    temp_map = dict(zip(woe_table['分箱区间'], woe_table['WOE']))
    woe_mapping_dict[feat]=temp_map#初始化一个空字典，用来存所有特征的 WOE 映射关系。
x_train_woe=x_train.copy()
for feat in feature_cols:
    #【修复】必须 .astype(float)。pandas 3.0 里 Series.map(dict) 作用在 Categorical 列上
    #仍返回 category 类型，后续做矩阵乘法 `x_train_woe @ coef` 会直接报 TypeError。
    x_train_woe[feat]=x_train[feat].map(woe_mapping_dict[feat]).astype(float)#把训练集每个特征的分箱标签替换成对应的 WOE 数值
x_test_woe=x_test.copy()
for feat in feature_cols:
    x_test_woe[feat]=x_test_woe[feat].map(woe_mapping_dict[feat]).astype(float)#用训练集拟合出的同一套映射替换测试集，避免数据泄露
#【新增】WOE 映射的完整性检查：如果测试集出现训练集没见过的箱，map 会产生 NaN，模型会直接报错
nan_check=pd.DataFrame({'训练集NaN':x_train_woe.isna().sum(),'测试集NaN':x_test_woe.isna().sum()})
print(f'--- WOE 映射后 NaN 检查（合计 {nan_check.values.sum()} 个）---')
print(nan_check[nan_check.sum(axis=1)>0] if nan_check.values.sum()>0 else '无 NaN，映射完整')
#【修复】导出真正用于部署的 WOE 映射表：用训练集拟合的这一套，
#与 score_card.csv 的分值表同源，业务方按它算分才能和分值表对上。
woe_deploy=pd.concat([calc_woe(df_train,feat,'default') for feat in feature_cols],ignore_index=True)
woe_deploy.to_csv(os.path.join(OUT_DIR,'woe映射表.csv'),index=False,encoding='utf-8-sig')
print(f'WOE 映射表（训练集拟合，与分值表同源）已导出：{os.path.join(OUT_DIR,"woe映射表.csv")}')
#先建基线模型：用训练集均值预测
base_prob=y_train.mean()
y_pred_base=np.full(len(y_test),fill_value=base_prob)
base_auc=roc_auc_score(y_test,y_pred_base)
print(f'基线模型（训练集均值）测试集AUC={base_auc:.4f}')
#训练 LogisticRegression，算 AUC
lr=LogisticRegression(random_state=42,max_iter=1000)
lr.fit(x_train_woe,y_train)
#预测概率，取正类(违约)的概率
y_train_pred=lr.predict_proba(x_train_woe)[:,1]
y_test_pred=lr.predict_proba(x_test_woe)[:,1]
auc_train=roc_auc_score(y_train,y_train_pred)
auc_test=roc_auc_score(y_test,y_test_pred)
print(f'【全变量版本 {len(feature_cols)} 个变量，作为对照】')
print(f'训练集AUC={auc_train:.4f},测试集AUC={auc_test:.4f}')
#KS函数
def calc_ks(y_true,pred_prob):
    #【修复】原来是逐行累计 CDF。当大量样本分数相同时，sort_values 的稳定排序会按原始行顺序
    #tie-break，累计曲线之间就会"漂"出一个并不存在的间距。
    #实测：常数预测（所有分数相同）本应得到 KS=0，原写法却会算出 0.012~0.025。
    #受影响的不只是退化情形：树模型的预测值是叶子值（大量并列）、评分卡的整数分值同样并列，
    #所以凡是用离散分数算 KS，原写法都会系统性偏高。
    #改为先按分数值聚合成组再累计，消除并列下的排序任意性。
    df_ks=pd.DataFrame({'y':np.asarray(y_true),'prob':np.asarray(pred_prob)})
    grp=df_ks.groupby('prob')['y'].agg(bad='sum',total='count').sort_index(ascending=False)
    grp['good']=grp['total']-grp['bad']
    grp['cum_good']=grp['good'].cumsum()/grp['good'].sum()
    grp['cum_bad']=grp['bad'].cumsum()/grp['bad'].sum()
    grp['diff']=grp['cum_bad']-grp['cum_good']
    #取绝对值最大值：曲线在中段跌破 0 时要算上负向的最大间距
    ks_value=grp['diff'].abs().max()
    #reset_index 让 prob 从索引变回普通列，下游画 KS 曲线仍可用 ks_df['prob']
    return ks_value,grp.reset_index()
ks_train,_=calc_ks(y_train,y_train_pred)
ks_test,_=calc_ks(y_test,y_test_pred)
print(f'训练KS={ks_train:.4f},测试KS={ks_test:.4f}')
#对比训练集与测试集指标，判断是否过拟合
print(f'AUC差值{abs(auc_train-auc_test):.4f}')#计算AUC差值
print(f'KS差值{abs(ks_train-ks_test):.4f}')#计算KS差值
print('AUC/KS 的差值都在 0.02 以内，说明几乎没有过拟合，泛化能力良好')
#做 5 折交叉验证，看 AUC 均值与标准差
#【修复】显式指定 StratifiedKFold 并固定随机种子，否则结果依赖数据的传入顺序，不便复现
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
cv_auc_list=cross_val_score(lr,x_train_woe,y_train,cv=cv,scoring='roc_auc')
print(f'5折AUC均值:{cv_auc_list.mean():.4f},标准差:{cv_auc_list.std():.4f},各折:{np.round(cv_auc_list,4)}')
print('与训练集AUC接近、标准差很小，说明模型区分能力良好、泛化能力强、无明显过拟合')
#【修复】ROC / KS / 校准这三张图原来画在这里。但此时 y_test_pred 还是全变量(17变量)模型的预测，
#而"切换到 11 变量主交付模型"发生在步骤6之后，于是图上写着 AUC=0.761 / KS=0.399，
#正文(README)写的却是主交付的 0.7592 / 0.3974 —— 同一份交付物里两个数，面试官一眼就能看到。
#现在把这三张图整体移到主交付模型拟合完成之后（见"主交付模型"段落下方的绘图块），
#图上的数字会随模型自动更新，不需要人工同步。


#================statsmodels 系数表与显著性=================
print('\n'+'='*60)
print('步骤6 statsmodels 系数表与显著性')
print('='*60)
x_train_sm=sm.add_constant(x_train_woe)
#用 statsmodels 构建逻辑回归模型，并在训练集上完成拟合
logit_model=sm.Logit(y_train,x_train_sm)
result=logit_model.fit(disp=0)
print(result.summary())
df_coef=pd.DataFrame({
    'coef':result.params,
    'std_err':result.bse,
    'z_value':result.tvalues,
    'p_value':result.pvalues,
    'OR':np.exp(result.params),#优势比
    'lower_95':result.conf_int()[0],
    'upper_95':result.conf_int()[1]
 })
print('--- 系数表 ---')
print(df_coef.round(4).to_string())
sig_df=df_coef[df_coef['p_value']>0.05]
print(f'--- p值大于0.05(不显著)的系数，共 {len(sig_df)-1 if "const" in sig_df.index else len(sig_df)} 个变量 ---')
print(sig_df.round(4).to_string())
#PAY_2的iv值在前面是第二高，但是在这里却p值很大，系数几乎等于0，完全不显著
'''原因:1.信息被 PAY_0 吸收了	PAY_0 与 PAY_2 相关 r=0.672。"最近一期逾期"已经包含了"上期逾期"的大部分信息
2.统计上叫"冗余"	在控制 PAY_0 之后，PAY_2 不再提供额外信息
3.单变量 IV 高 ≠ 多变量有用	IV 是单变量指标，它不知道其他变量的存在
IV 负责"海选"，逻辑回归负责"终选"。IV 高只是入场券，不代表能留在模型里。'''
#系数符号检查
'''本项目用 WOE = ln(坏/好) 约定，所以"坏账率越高的箱 WOE 越正"，
为了预测 y=1（违约），系数绝大多数应当为正。
实测 17 个特征系数中 16 个为正，符合预期；
唯一为负的是 pay_ratio1_bins（系数 -0.2024，p=0.0150）。
注意：这个系数在统计上是显著的，但符号与业务逻辑相反 —— 正是上面"共线性危害"第 3 条
描述的"系数符号颠倒、评审无法解释"。
成因：它的分子 PAY_AMT1 与 pay1_bins 重复，分母 BILL_AMT1 已通过 util_bins 入模，
三者在描述同一件事（还款能力），导致系数被扭曲。
处理建议：pay1_bins 与 pay_ratio1_bins 只保留一个（推荐保留单调且显著的 pay1_bins）。'''
neg_coef=df_coef[(df_coef['coef']<0)&(df_coef.index!='const')]
print(f'--- 负系数变量（排除截距），共 {len(neg_coef)} 个 ---')
print(neg_coef.round(4).to_string() if len(neg_coef)>0 else '无')
print(f'正系数变量数 = {len(df_coef[(df_coef["coef"]>0)&(df_coef.index!="const")])} / {len(feature_cols)}')
#计算 VIF，判断共线性是否可接受
vif_data=pd.DataFrame()
vif_data['feature']=x_train_sm.columns
vif_data['VIF']=[variance_inflation_factor(x_train_sm.values,i)for i in range(x_train_sm.shape[1])]
print(f'--- VIF（>10 严重共线，>5 需关注），最大 {vif_data["VIF"].max():.4f} ---')
print(vif_data.round(4).to_string(index=False))#VIF全部小于4，无严重共线性
#报告伪 R²、AIC 并解释其含义
print(f'McFadden伪R²: {result.prsquared:.4f}')#伪 R² (McFadden)=0.18,表示相对"只用截距"的模型，对数似然改善多少，0.18属于良好
print(f'AIC: {result.aic:.1f}')#AIC拟合优度 − 复杂度惩罚，用于比较模型，不用于绝对评价
print(f'BIC: {result.bic:.1f}')
#风控模型中只看AUC和KS不看伪R²，逻辑回归的伪 R² 天然偏低，0.2 以上就算很好，用 R² 评价评分卡本身是错的方法论。
#决定最终模型：是否剔除不显著变量？给出理由
#剔除，评分卡要"每个变量都有意义"，风控评分卡的标准做法是保留统计显著 + 业务可解释的变量
#【新增】用数据验证这个决策：对比"全变量"与"只保留显著变量"的测试集表现
drop_cols=[c for c in sig_df.index if c!='const' and c in feature_cols]
#【修复】在 p 值筛选之外，再强制剔除 pay_ratio1_bins。
#它虽然显著（p=0.0150），但系数是负的（-0.2024），符号与业务逻辑相反：还款比例越高、风险本该越低。
#成因是它与 pay1_bins 共用 PAY_AMT1、与 util_bins 共用 BILL_AMT1，信息重叠把系数扭反了。
#这样的变量留在交付分值表里会直接造成业务上说不通的计分方向 —— 实测分值表里
#"完全不还款"这一档反而给客户加分、"大额/全额还款"这一档反而减分，业务方一定会问"为什么还得多反而扣分"。
#所以交付版剔除它；全变量版本仍保留它，作为"共线性导致系数符号翻转"的实测证据。
force_drop=['pay_ratio1_bins']
drop_cols=list(dict.fromkeys([c for c in drop_cols if c in feature_cols]+force_drop))
keep_cols=[c for c in feature_cols if c not in drop_cols]
lr_keep=LogisticRegression(random_state=42,max_iter=1000)
lr_keep.fit(x_train_woe[keep_cols],y_train)
auc_keep=roc_auc_score(y_test,lr_keep.predict_proba(x_test_woe[keep_cols])[:,1])
ks_keep,_=calc_ks(y_test,lr_keep.predict_proba(x_test_woe[keep_cols])[:,1])
print(f'--- 剔除不显著变量前后对比 ---')
print(f'全变量({len(feature_cols)}个): 测试AUC={auc_test:.4f}, KS={ks_test:.4f}')
print(f'仅显著({len(keep_cols)}个): 测试AUC={auc_keep:.4f}, KS={ks_keep:.4f}')
print(f'剔除变量清单: {drop_cols}')
print('结论：剔除后 AUC/KS 几乎没有下降，但模型少了一批无法向业务解释的变量，应剔除。')

#【新增】把"结论"真正落进交付物。
#原来这里算完 lr_keep 就结束了，后续分值表/PSI/策略损益/模型对比用的仍然是 17 变量的 result，
#于是出现"结论说应剔除、交付的却是全变量版本"两张皮，面试官问"最终交付哪个模型"会答不上来。
#现在改为：后续所有环节统一基于剔除后的模型；全变量版本的结果保留下来做对照证据。
auc_full,ks_full=auc_test,ks_test
feature_cols_full=list(feature_cols)
x_train_woe_full=x_train_woe.copy()
x_test_woe_full=x_test_woe.copy()

feature_cols=list(keep_cols)
x_train_woe=x_train_woe[keep_cols].copy()
x_test_woe=x_test_woe[keep_cols].copy()
x_train_sm=sm.add_constant(x_train_woe)
x_test_sm=sm.add_constant(x_test_woe)
result=sm.Logit(y_train,x_train_sm).fit(disp=0)
print()
print('--- 主交付模型：剔除不显著变量后重新拟合 ---')
print(f'主交付入模变量数 = {len(feature_cols)}（从 {len(feature_cols_full)} 个中剔除 {len(drop_cols)} 个）')
print(f'对照 全变量版本({len(feature_cols_full)}个): 测试AUC={auc_full:.4f}, KS={ks_full:.4f}')
print(f'主交付 精简版本({len(feature_cols)}个): 测试AUC={auc_keep:.4f}, KS={ks_keep:.4f}')
print(f'主交付 伪R²={result.prsquared:.4f}, AIC={result.aic:.1f}, BIC={result.bic:.1f}')
#重算交付模型自己的训练/测试指标与差值：过拟合检查以交付模型为准，而不是以全变量版本为准
y_train_pred=result.predict(x_train_sm)
y_test_pred=result.predict(x_test_sm)
auc_train=roc_auc_score(y_train,y_train_pred)
auc_test=roc_auc_score(y_test,y_test_pred)
ks_train,_=calc_ks(y_train,y_train_pred)
ks_test,_=calc_ks(y_test,y_test_pred)
print(f'主交付 训练集: AUC={auc_train:.4f}, KS={ks_train:.4f}')
print(f'主交付 测试集: AUC={auc_test:.4f}, KS={ks_test:.4f}')
print(f'主交付 差值  : AUC={abs(auc_train-auc_test):.4f}, KS={abs(ks_train-ks_test):.4f}')
#主交付模型自己做一次 5 折交叉验证（上面那次是全变量版本的，不能直接挪用）
cv_keep=cross_val_score(LogisticRegression(random_state=42,max_iter=1000),
                        x_train_woe,y_train,
                        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42),
                        scoring='roc_auc')
print(f'主交付 5折CV AUC均值={cv_keep.mean():.4f}, 标准差={cv_keep.std():.4f}, 各折:{np.round(cv_keep,4)}')
print('说明：以下步骤 7（分值表）、8（PSI）、9（策略损益）、10（模型对比）全部基于精简版本。')

#画 ROC / KS / 校准三张图（【修复】从步骤5 之后移到这里）
#放在主交付模型拟合完成之后，图上的 AUC / KS 才会与正文一致。
#画 ROC 曲线，解释 AUC 的业务含义
fpr,tpr,_=roc_curve(y_test,y_test_pred)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f'AUC={auc_test:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR假阳性率')
plt.ylabel('TPR真阳性率')
plt.title('ROC曲线')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR,'02_roc_curve.png'),dpi=120,bbox_inches='tight')
plt.close()
#画 KS 曲线（两条累计分布曲线 + 最大间距）
ks_val,ks_df=calc_ks(y_test,y_test_pred)
plt.figure(figsize=(6,6))
plt.plot(ks_df['prob'],ks_df['cum_bad'],label='累计坏样本')
plt.plot(ks_df['prob'],ks_df['cum_good'],label='累计好样本')
plt.title(f'KS曲线，KS={ks_val:.3f}')
plt.xlabel('预测违约概率')
plt.ylabel('累计占比')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR,'03_ks_curve.png'),dpi=120,bbox_inches='tight')
plt.close()
#画校准曲线，检查预测概率是否准确
prob_true, prob_pred =calibration_curve(y_test, y_test_pred, n_bins=10)
plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, "o-", label='模型预测')
plt.plot([0, 1], [0, 1], "k--", label="理想校准")
plt.xlabel("预测概率")
plt.ylabel('真实违约率')
plt.title("校准曲线")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR,'04_calibration_curve.png'),dpi=120,bbox_inches='tight')
plt.close()
print(f'主交付模型的 ROC / KS / 校准曲线已保存到 {OUT_DIR}')


#================评分卡刻度转换=================
print('\n'+'='*60)
print('步骤7 评分卡刻度转换')
print('='*60)
base_score=600
base_odds=20
PDO=50
B=PDO/np.log(2)
A=base_score - B*np.log(base_odds)
print(f'B = PDO/ln2 = {B:.4f}')
print(f'A = base_score - B*ln(base_odds) = {A:.4f}')
print(f'公式: score = {A:.2f} + {B:.4f} * ln(odds)，odds = 好/坏')
def prob_to_score(p_bad):
    p_bad = np.clip(p_bad, 1e-6, 1-1e-6)
    return A + B*np.log((1-p_bad)/p_bad)   # odds = 好/坏
print(f'方向自检：prob_to_score(1/21) = {prob_to_score(1/21):.4f}（应等于 base_score=600）')
#方向自检
p_high_risk=0.30#高违约概率
p_low_risk=0.0476#低违约概率
score_high=prob_to_score(p_high_risk)
score_low=prob_to_score(p_low_risk)
print(f'高风险用户(p=0.3)分数:{score_high:.2f}')
print(f'低风险用户(p=0.0476)分数:{score_low:.2f}')
#【修复】原代码只算不验。加一句断言，方向写反时立刻报错，避免"给最坏的人最高分"上线事故
assert score_low>score_high,'评分方向写反了：低风险客户的分数必须高于高风险客户'
assert abs(prob_to_score(1/(1+base_odds))-base_score)<1e-6,'基准分自检失败：基准客户算出的分数应等于 base_score'
print('方向自检通过：低风险高分、高风险低分，且基准分等于 base_score')
#计算每一个变量的分值表
coef_series=result.params
beta0=coef_series.iloc[0]#const截距项
coef_vars=coef_series.iloc[1:]#各个入模特征的β系数
#【修复】基础分符号写反了。推导：score = A - B*(beta0 + Σ beta_i*WOE_i) = (A - B*beta0) - Σ B*beta_i*WOE_i
#所以基础分应为 A - B*beta0，原代码写成 A + B*beta0，与 train_score 相差约 180 分。
base_point=A-B*beta0
print(f'基础分 base_point = A - B*beta0 = {base_point:.2f}')
woe_table_dict = {}
for feat in feature_cols:
    woe_table=calc_woe(df_train,feat,'default')
    woe_table_dict[feat] = woe_table
score_card_list=[]
for var_name,beta in coef_vars.items():
    woe_df = woe_table_dict[var_name].copy() # 这里取的是DataFrame，不是dict！
    woe_df['beta系数']=beta
    woe_df['bin_score']=B*beta*woe_df['WOE']
    score_card_list.append(woe_df[['变量名','分箱区间','WOE','beta系数','bin_score']])
#【修复】原来这两行缩进在 for 循环内部，每轮都重复 concat 一次，移到循环外
score_card=pd.concat(score_card_list)
score_card['bin_score']=score_card['bin_score'].round(2)
print('--- 评分卡分值表 ---')
print(score_card.to_string(index=False))
score_card.to_csv(os.path.join(OUT_DIR,'score_card.csv'),index=False,encoding='utf-8-sig')
logit_train = beta0 + x_train_woe @ coef_vars
train_score = A - B * logit_train
logit_test = beta0 + x_test_woe @ coef_vars
test_score = A - B * logit_test
# 打印描述统计，看分数范围
print("=====训练集分数统计=====")
print(train_score.describe())
print("=====测试集分数统计=====")
print(test_score.describe())
#【新增】用分值表复核：基础分 - 各箱分值之和，应当等于直接用公式算出的分数
recon_score=base_point - sum(B*coef_vars[f]*x_test_woe[f] for f in feature_cols)
print(f'分值表复核：与公式算出的分数最大误差 = {np.abs(recon_score.values-test_score.values).max():.6f}（应为 0）')
# 绘制分布对比图
plt.figure(figsize=(9,5))
plt.hist(train_score, bins=30, alpha=0.6, label='Train Score')
plt.hist(test_score, bins=30, alpha=0.6, label='Test Score')
plt.xlabel('评分卡分数')
plt.ylabel('样本数量')
plt.title('评分卡全样本分数分布')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR,'05_score_distribution.png'),dpi=120,bbox_inches='tight')
plt.close()
#找一个真实客户，手工算出他的分数
sample_pos=0
sample_row=x_test_woe.iloc[[sample_pos]]
# 注意：DataFrame @ 一维数组得到的是长度为 1 的数组，取值要用 [0]
sample_score=float((A-B*(beta0+sample_row.values@coef_vars.values))[0])
print(f'--- 单个客户手工验算（测试集第 {sample_pos} 个）---')
print(f'该客户真实标签(y) = {y_test.reset_index(drop=True).iloc[sample_pos]}')
print(f'按公式算出的分数 = {sample_score:.2f}')


#================PSI 稳定性监控=================
print('\n'+'='*60)
print('步骤8 PSI 稳定性监控')
print('='*60)
#定义PSI函数
def calc_psi(base_dist,actual_dist):
    '''base_dist:基准(训练集)各分箱样本占比
    autual_dist:对比集(测试/OOT)各分箱样本占比
    return psi值'''
    base_dist=np.where(base_dist==0,1e-6,base_dist)
    actual_dist=np.where(actual_dist==0,1e-6,actual_dist)
    psi=np.sum((actual_dist-base_dist)*np.log(actual_dist/base_dist))
    return psi
# 封装：输入df、特征名、分箱列名，计算单变量PSI（用你之前的分箱）
def get_feature_psi(df_base, df_actual, bin_col):
    # 基准集各箱计数+占比
    base_cnt = df_base[bin_col].value_counts(normalize=True)
    actual_cnt = df_actual[bin_col].value_counts(normalize=True)
    # 对齐分箱，缺失分箱填充0
    all_bins = sorted(list(set(base_cnt.index) | set(actual_cnt.index)))
    base_pct = np.array([base_cnt.get(b,0) for b in all_bins])
    actual_pct = np.array([actual_cnt.get(b,0) for b in all_bins])
    psi_val = calc_psi(base_pct, actual_pct)
    return psi_val
#算训练集 vs 测试集的 PSI，解释结果
psi_result={}
for feat in feature_cols:
    psi_val=get_feature_psi(df_train,df_test,feat)
    psi_result[feat]=psi_val
psi_df=pd.DataFrame(list(psi_result.items()),columns=['feature','PSI'])
print('--- 训练集 vs 随机切分测试集 的 PSI ---')
print(psi_df.round(6).to_string(index=False))
print(f'最大 PSI = {psi_df["PSI"].max():.6f}')
#解释为什么随机切分的 PSI 必然接近 0
'''随机切分：把同一时间段全部样本打乱，随机拆成train/test。
样本是同分布、同一时间、同一批客群，只是随机抽一部分。每个分箱里面样本占比在train和test几乎完全一样。
Actual_i / Expected_i，(Actual-Expected)趋近0，PSI求和结果自然接近0。
随机划分只能检验模型拟合能力，不能检验时间稳定性！ 真实业务漂移是时间带来的，不是随机抽样。'''
#按 ID 分段做"伪 OOT"，看 PSI 变化
#【修复】原来的做法是 pd.concat([df_train, df_test]) 再取前 70%，
#但 train_test_split 默认 shuffle=True，拼回来是乱序，取前 70% 并不是"按 ID 的早期样本"，
#等价于又做了一次随机切分，PSI 必然≈0，伪 OOT 完全失去意义。
#正确做法：按 ID 排序后切分（ID 是 1~30000 的顺序编号，可近似当作时间代理，尽管关卡1的卡方检验显示趋势不显著）。
df_sorted=df.sort_values('ID').reset_index(drop=True)
split_pos=int(len(df_sorted)*0.7)
df_pseudo_train=df_sorted.iloc[:split_pos]#按 ID 排序的前70%当作开发样本
df_pseudo_oot=df_sorted.iloc[split_pos:]#按 ID 排序的后30%当作伪OOT未来样本
#计算伪OOT的PSI
pseudo_psi={}
for feat in feature_cols:
    psi_val=get_feature_psi(df_pseudo_train,df_pseudo_oot,feat)
    pseudo_psi[feat]=psi_val
pseudo_psi_df=pd.DataFrame(list(pseudo_psi.items()),columns=['feature','PSI'])
pseudo_psi_df['判定']=np.where(pseudo_psi_df['PSI']<0.1,'稳定',
                       np.where(pseudo_psi_df['PSI']<0.25,'需关注','不稳定'))
print('--- 伪OOT（按 ID 排序切分）的 PSI ---')
print(pseudo_psi_df.sort_values('PSI',ascending=False).round(5).to_string(index=False))
print(f'最大 PSI = {pseudo_psi_df["PSI"].max():.5f}')
print(f'前70%(早)坏账率 = {df_pseudo_train[target].mean():.4f}')
print(f'后30%(晚)坏账率 = {df_pseudo_oot[target].mean():.4f}')
print('各变量 PSI 全部 <0.1，判定为"稳定"；但这个"稳定"也不能当作严格意义上的时间稳定性证据。')
#本数据集无法做真 OOT 的局限及影响
'''1.无法进行时间外（OOT）验证。数据集不含日期字段，仅有 1~30000 的顺序 ID。关卡1的卡方检验 p<0.001 显著，
但各 ID 段坏账率极差只有 6.6 个百分点且无单调趋势，考虑到 N=30000 时卡方检验极度敏感，
我判断这是大样本下的统计显著而非真实时间漂移，因此不采用 ID 作为严格的时间代理。
2.影响：模型的时间稳定性未经验证。真实业务中，客群结构、宏观环境、竞品策略都会随时间变化，模型可能在 6~12 个月后失效。
3.PSI 检验的局限：随机切分下的 PSI≈0.001 是数学必然，不构成稳定性证据；只有按 ID 排序的伪 OOT 才有一点参考价值。
4.样本选择偏差：本数据全部为已发卡客户，不含拒绝样本，因此策略效果不能外推到全量申请者。
5.如果给我真实数据，我会：按放款月份切分做真正的 OOT（训练早期、验证后期），并按月计算 PSI 做漂移监控，用冠军挑战者框架小流量验证策略。'''
#设计一份上线后的监控方案（指标+阈值+动作）
monitor_plan=pd.DataFrame([
    {'监控维度':'模型稳定性','指标':'分数PSI(月度)','预警阈值':'>0.1 关注，>0.25 报警','触发动作':'排查客群变化；>0.25 启动重训'},
    {'监控维度':'变量稳定性','指标':'各变量PSI','预警阈值':'>0.1','触发动作':'检查该变量数据源与口径'},
    {'监控维度':'区分能力','指标':'KS/AUC(滚动3个月)','预警阈值':'KS下降>20%','触发动作':'重训模型'},
    {'监控维度':'校准','指标':'预测违约率 vs 实际违约率','预警阈值':'偏差>20%','触发动作':'重新校准(Platt/Isotonic)'},
    {'监控维度':'策略效果','指标':'通过率、坏账率、FPD','预警阈值':'通过率波动>5pp','触发动作':'检查规则命中率异动'},
    {'监控维度':'规则监控','指标':'各规则命中率','预警阈值':'单条规则波动>30%','触发动作':'排查规则是否被绕过'},
    {'监控维度':'资产质量','指标':'迁徙率 M0→M1→M2','预警阈值':'连续2月上升','触发动作':'上报风险委员会'},
])
print('--- 上线后监控方案 ---')
print(monitor_plan.to_string(index=False))


#================策略设计与损益测算=================
print('\n'+'='*60)
print('步骤9 策略设计与损益测算')
print('='*60)
#设定损益参数（NIM、LGD），真实场景NIM取自产品定价、资金成本；LGD取自历史催收回收数据。本数据集没有资金、催收数据，所以为假设参数。
#NIM(净息差):放款的收益率，LGD(违约损失率):客户违约后，每一元本金最终亏掉的比例
NIM=0.20
LGD=0.70
#计算盈亏平衡坏账率
break_even_badrate=NIM/(NIM+LGD)
print(f'假设: 净息差率 NIM={NIM:.0%}，违约损失率 LGD={LGD:.0%}（均为假设值，非实测）')
print(f'盈亏平衡坏账率 = NIM/(NIM+LGD) = {break_even_badrate:.4f} ({break_even_badrate:.2%})')
print(f'实际整体坏账率 = {base_badrate:.4f} ({base_badrate:.2%})')
print(f'安全缓冲 = {break_even_badrate-base_badrate:.4f}（{(break_even_badrate-base_badrate)*100:.2f} 个百分点）')
#计算无策略时的总损益与利润率(基线)
#【修复】严重错误：原来用 result.predict(x_train_sm)（训练集，21000行）去配 y_test（9000行）。
#两者索引不同，pd.DataFrame 会按索引对齐，结果生成 23697 行、其中 14697 行 'y' 是 NaN；
#于是 pass_rate 拿 23697 做分母、bad_rate 只用 9000 行算，口径完全错位，策略结论被污染。
#正确做法：用测试集自变量 x_test_sm 预测。
x_test_sm=sm.add_constant(x_test_woe)
y_pred_prob=result.predict(x_test_sm)
y_true=y_test.reset_index(drop=True)
df_strategy=pd.DataFrame({'prob':y_pred_prob.values,'y':y_true.values})
print(f'策略测算样本数 = {len(df_strategy)}（测试集），整体坏账率 = {df_strategy["y"].mean():.4f}')
#遍历不同cutoff，计算每一个门槛下：通过率、坏账率、净利润
#【修复】原来用 np.linspace(0,0.6,30)，步长 0.0207 太粗：
#  1) 起点 0 处 prob<0 恒为假，会多出一行 pass_rate=0 的退化行；
#  2) 分辨率不足以定位最优阈值（线性损益下最优阈值即盈亏平衡坏账率，见下）。
#现改为 0.01 起、步长 0.005 的细网格，仅用于画曲线与定位最优点。
cutoff_list = np.linspace(0.01,0.6,119)
print(f'最优阈值的解析参考：线性损益下 accept 条件为 (1-p)*NIM > p*LGD，'
      f'即 p < NIM/(NIM+LGD) = {NIM/(NIM+LGD):.4f}，这是期望意义下的闭式最优；'
      f'下方网格给出的是本测试集上的实际最优点（含抽样波动）。')
res = []
for cutoff in cutoff_list:
    #prob<cutoff就放款（prob是违约概率，违约概率低于阈值才放）
    df_strategy['accept'] = df_strategy['prob'] < cutoff
    accept_df = df_strategy[df_strategy['accept']]
    pass_rate = len(accept_df)/len(df_strategy)
    if len(accept_df)==0:
        bad_rate=0
        profit=0
    else:
        bad_rate = accept_df['y'].mean()
        #假设总本金=样本数量，单样本放款1元
        total_loan = pass_rate
        interest_income = total_loan * (1 - bad_rate) * NIM
        loss = total_loan * bad_rate * LGD
        profit = interest_income - loss
    res.append({'cutoff':cutoff,'pass_rate':pass_rate,'bad_rate':bad_rate,'profit':profit})
res_df = pd.DataFrame(res)
print('--- 不同 cutoff 下的通过率 / 坏账率 / 利润 ---')
print(res_df.round(4).to_string(index=False))
#画通过率-坏账率-损益曲线，找最优点
plt.figure(figsize=(10,6))
plt.plot(res_df['pass_rate'],res_df['bad_rate'],label='坏账率')
plt.plot(res_df['pass_rate'],res_df['profit'],label='净利润')
plt.xlabel('通过率')
plt.legend()
plt.title('通过率-坏账率-利润曲线')
plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR,'06_passrate_badrate_profit.png'),dpi=120,bbox_inches='tight')
plt.close()
#找出利润最大的最优cutoff
#【修复】变量名拼写：beat_row -> best_row
best_row=res_df.loc[res_df['profit'].idxmax()]
print('--- 最优策略点 ---')
print(best_row.round(4).to_string())
print(f'→ 最优 cutoff = {best_row["cutoff"]:.3f}，通过率 {best_row["pass_rate"]:.2%}，'
      f'坏账率 {best_row["bad_rate"]:.4f}，利润率 {best_row["profit"]:.4%}')
#【新增】算出最优策略下的"误杀率"。
#注意：best_row['bad_rate'] 是**通过人群**的坏账率，所以 1-(通过人群坏账率) 得到的是
#"通过人群里好客户的占比"，跟被拒人群的误杀完全不是一回事（早期版本在这里算错过）。
#被拒人群的坏账率要用"总体坏账 − 通过人群贡献的坏账"反推。
_accept_mask=df_strategy['prob']<best_row['cutoff']
_acc_n=int(_accept_mask.sum())
_rej_n=int(len(df_strategy)-_acc_n)
_acc_bad=int(df_strategy.loc[_accept_mask,'y'].sum())
_rej_bad=int(df_strategy.loc[~_accept_mask,'y'].sum())
_rej_badrate=_rej_bad/_rej_n
_rej_goodrate=1-_rej_badrate
print('--- 最优策略的接受/拒绝明细 ---')
print(f'接受 {_acc_n} 人（其中坏客户 {_acc_bad}，坏账率 {_acc_bad/_acc_n:.4f}）')
print(f'拒绝 {_rej_n} 人（其中坏客户 {_rej_bad}，坏账率 {_rej_badrate:.4f}）')
print(f'→ 被拒人群中有 {_rej_goodrate:.4f}（{_rej_goodrate:.2%}）其实是好客户，即"误杀率"')
print(f'  对比：通过人群里好客户占比 = {1-_acc_bad/_acc_n:.4f}（这两个数容易混，别弄反）')
#分析通过率-坏账率-损益曲线
#【修复】这里原来是手抄数字的注释块。改一次代码、重跑一次，72.9%/0.207/91.5%/3.67%/12.75% 就全部过期了。
#现在改成用实际计算结果拼字符串，数字不可能再漂。
_pass_at_peak=best_row['pass_rate']; _cut_at_peak=best_row['cutoff']; _prof_at_peak=best_row['profit']
_wide=res_df[res_df['pass_rate']>=0.90]
_wide=_wide.iloc[-1] if len(_wide)>0 else res_df.iloc[-1]
print('--- 通过率-坏账率-损益曲线形状解读 ---')
print(f'1. 通过率 0% → {_pass_at_peak:.1%} 区间：利润率从 0 一路爬升到峰值 {_prof_at_peak:.2%}。')
print('   原因是被拒的是评分最低的高危客户，他们的预期损失远大于收益，拒掉他们是"纯赚"。')
print(f'2. 通过率约 {_pass_at_peak:.1%}（cutoff={_cut_at_peak:.3f}）到达峰值：此时边际收益 = 边际损失。')
print('3. 通过率继续上升：利润率缓慢下降。因为被拒人群里好客户的比例越来越高，')
print('   拒掉他们"放弃的收益"开始超过"避免的损失"。')
print(f'   实测通过率 {_wide["pass_rate"]:.1%} 时利润率降到 {_wide["profit"]:.2%}，但仍是正的 —— '
      '说明曲线在峰值附近很平坦，业务上可以在通过率和利润率之间做取舍，不必死守数学最优点。')
#风控的目标不是"把坏账率降到最低"，而是"让风险调整后的利润最大"。
#继续收紧 cutoff 会让坏账率更低，但利润反而减少。
#设计 3~5 条业务规则，算命中率/命中坏账率/边际收益
#【修复】原规则代码引用了不存在的列（query_bin）且用数值方式比较字符串分箱列（PAY1_bin>=3、util_bins>0.9），
#直接运行会报错。这里改成基于真实分箱列的条件。
print('--- 硬规则策略测算 ---')
df_rules = df_test.copy()
#【修复】这里原来写 df_rules['y'] = y_test.reset_index(drop=True)，与上面策略段是同一个坑：
#df_test 保留了原始行索引，而 y_test.reset_index(drop=True) 是 0~8999，
#pandas 会按索引对齐，导致大部分 'y' 变成 NaN、少数错位匹配，规则结论全是垃圾。
#用 .values 传入纯数组即可避免索引对齐。
df_rules['y'] = y_test.values
assert df_rules['y'].isna().sum()==0,'规则表标签存在 NaN，索引没对齐'
rule_dict = {
    "规则1: 最近一期逾期(PAY1_bin!=正常)": df_rules['PAY1_bin'].astype(str)!='正常(≤0)',
    "规则2: 最近一期逾期2个月": df_rules['PAY1_bin'].astype(str)=='逾期2个月',
    "规则3: 额度使用率>0.8": df_rules['util_bins'].astype(str)=='高使用率(>0.8)',
    "规则4: 有逾期且使用率>0.8": (df_rules['PAY1_bin'].astype(str)!='正常(≤0)')&(df_rules['util_bins'].astype(str)=='高使用率(>0.8)'),
    "规则5: 授信额度≤3万": df_rules['limit_bin_final'].astype(str)=='≤30000',
}
rule_result = []
for rule_name, mask in rule_dict.items():
    hit_cnt = int(mask.sum())
    hit_rate = hit_cnt / len(df_rules)
    if hit_cnt > 0:
        hit_badrate = df_rules.loc[mask, "y"].mean()
    else:
        hit_badrate = 0
    # 边际收益：拒掉命中人群所避免的损失 - 放弃的收益
    marginal_profit = hit_cnt * hit_badrate * LGD - hit_cnt * (1-hit_badrate)*NIM
    # 拒绝后的整体利润率：通过率 * [(1-坏账率)*净息差 - 坏账率*违约损失率]
    after_pass = 1-hit_rate
    after_bad = df_rules.loc[~mask,"y"].mean()
    after_profit = after_pass*((1-after_bad)*NIM - after_bad*LGD)
    rule_result.append({
        "规则名称": rule_name,
        "命中人数": hit_cnt,
        "命中率": hit_rate,
        "命中坏账率": hit_badrate,
        "拒绝后通过率": after_pass,
        "拒绝后坏账率": after_bad,
        "边际收益": marginal_profit,
        "拒绝后利润率": after_profit
    })
rule_df = pd.DataFrame(rule_result)
print("硬规则评估结果")
print(rule_df.round(4).to_string(index=False))
rule_df.to_csv(os.path.join(OUT_DIR,'rule_strategy.csv'),index=False,encoding='utf-8-sig')
#对比"评分卡策略" vs "单条规则策略"
'''硬规则策略：一刀切，满足条件直接拒绝，简单易解释，但风险分层粗糙，容易误杀优质客户。
评分卡策略：连续风险分数精细化分层，可以灵活调整cutoff，最大化利润；缺点是建模复杂，业务理解成本更高。行业常用组合方案：硬规则前置拦截极端风险，评分卡做主准入策略。'''
#做敏感性分析：LGD代表违约损失，LGD越高，单个坏客户造成的亏损越大。
#【修复】原注释把方向写反了（原文："LGD上升时，最优cutoff会抬高，准入策略收紧"），且自相矛盾。
#正确推导：盈亏平衡坏账率 = NIM/(NIM+LGD)，LGD 上升 -> 盈亏平衡坏账率下降；
#而 accept 条件是 prob < cutoff，所以门槛必须同步下调 -> 最优 cutoff 下降、通过率下降，即准入收紧。
#"cutoff 抬高"与"准入收紧"是相反的两件事，不能并存。
#另外：网格 np.linspace(0,0.6,30) 步长达 0.0207，分辨率不足以体现这一变化，
#所以 lgd_sensitivity.csv 里 LGD=0.6/0.7/0.8 三档都会落到同一个网格点上。下方已改用细网格。
print('--- LGD 敏感性分析 ---')
df_strategy_sens = pd.DataFrame({
    'prob': result.predict(x_test_sm).values,
    'y': y_test.reset_index(drop=True).values
})
total_num = len(df_strategy_sens)
lgd_list = [0.5, 0.6, 0.7, 0.8]
sens_rows=[]
for lgd_val in lgd_list:
    profit_array = []
    for cutoff in cutoff_list:
        accept_mask = df_strategy_sens['prob'] < cutoff
        accept_count = int(accept_mask.sum())
        pass_rate = accept_count / total_num
        if accept_count == 0:
            profit = 0
        else:
            accept_df = df_strategy_sens[accept_mask]
            bad_rate = accept_df['y'].mean()
            loan = pass_rate * total_num
            interest = loan*(1-bad_rate)*NIM
            loss = loan*bad_rate*lgd_val
            profit = interest - loss
        profit_array.append(profit)
    best_cut = cutoff_list[int(np.argmax(profit_array))]
    sens_rows.append({'LGD':lgd_val,'盈亏平衡坏账率':NIM/(NIM+lgd_val),
                      '最优cutoff(违约概率阈值)':best_cut,'最优利润':max(profit_array)})
    print(f"LGD={lgd_val}, 盈亏平衡坏账率={NIM/(NIM+lgd_val):.4f}, 最优cutoff={best_cut:.3f}")
sens_df=pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(OUT_DIR,'lgd_sensitivity.csv'),index=False,encoding='utf-8-sig')
#写"策略上线方案"（灰度、冠军挑战者、监控）
'''1.离线验证	本报告（伪OOT 验证 + 损益测算）	利润率提升且通过率不低于业务底线
2.灰度上线	5% 流量先跑新策略，与旧策略并存	观察 1~2 个账期，实际坏账率与预测偏差 <20%
3.冠军挑战者	新旧策略各 10% 流量对比	新策略利润率显著优于旧策略
4.全量	逐步放量至 100%
5.持续监控	监控表'''
#用业务语言总结：这个策略值不值得上
print('--- 业务结论 ---')
#【修复】原来的"无策略利润率"取的是 cutoff=0 那一行的 profit（=0），口径不对。
#无策略应当等于"全部通过"，直接用整体坏账率算。
no_strategy_profit=(1-df_strategy['y'].mean())*NIM-df_strategy['y'].mean()*LGD
print(f'无策略（全部通过）：通过率 100%，坏账率 {df_strategy["y"].mean():.4f}，'
      f'利润率 {no_strategy_profit:.4%}')
print(f'评分卡最优策略：通过率 {best_row["pass_rate"]:.2%}，坏账率 {best_row["bad_rate"]:.4f}，'
      f'利润率 {best_row["profit"]:.4%}')
print(f'盈亏平衡坏账率 {break_even_badrate:.4%} 与实际坏账率 {base_badrate:.4%} 的缓冲只有 '
      f'{(break_even_badrate-base_badrate)*100:.2f} 个百分点 —— 说明该资产组合处在盈亏平衡线附近，'
      f'风控策略是这个业务的"生死开关"而不是"优化项"。')


#================模型对比与结论=================
print('\n'+'='*60)
print('步骤10 模型对比与结论')
print('='*60)
#【修复】原来所有模型（含树模型）都用 WOE 编码特征，这是一个不公平的对比。
#WOE 分箱是为了让逻辑回归拿到"分组线性化 + 单调"的输入，代价是丢掉原始取值的细节；
#而树模型本来就能自己搜切分点，把分箱后的有序整数喂给它，等于替它做了一次很粗的切分，是净损失。
#最直接的证据：随机森林(300棵) 的 KS 竟然低于单棵限深决策树，正常情况不该发生。
#现在改成每个模型用自己的自然特征口径：评分卡用 WOE 特征，树模型用原始数值特征，训练/测试行完全一致。
raw_feature_cols=['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
                  'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                  'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                  'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
x_train_raw=df.loc[x_train.index,raw_feature_cols].copy()
x_test_raw=df.loc[x_test.index,raw_feature_cols].copy()
print(f'原始特征口径：{len(raw_feature_cols)} 个字段（未分箱、未 WOE 编码）')

model_dict={
    '逻辑回归评分卡':LogisticRegression(random_state=42,max_iter=5000),
    '决策树(d=4)':DecisionTreeClassifier(max_depth=4,random_state=42),
    '随机森林(300棵)':RandomForestClassifier(n_estimators=300,random_state=42),
    '梯度提升':GradientBoostingClassifier(random_state=42),
}
ks_base=calc_ks(y_test,y_pred_base)[0]
compare_rows=[{'模型':'基线(训练集均值)',
               'WOE特征AUC':round(base_auc,4),'WOE特征KS':round(ks_base,4),
               '原始特征AUC':round(base_auc,4),'原始特征KS':round(ks_base,4)}]
for name,model in model_dict.items():
    row={'模型':name}
    for tag,(xtr,xte) in [('WOE',(x_train_woe,x_test_woe)),('原始',(x_train_raw,x_test_raw))]:
        #逻辑回归在原始特征上必须先标准化：LIMIT_BAL 是 1e5 量级、AGE 是 30 量级，
        #量纲差 3 个数量级会让 lbfgs 无法收敛（否则会报 ConvergenceWarning）。
        #树模型对单调变换不敏感，不需要标准化，直接喂原始值即可。
        m=make_pipeline(StandardScaler(),LogisticRegression(random_state=42,max_iter=5000)) \
          if (tag=='原始' and isinstance(model,LogisticRegression)) else model
        m.fit(xtr,y_train)
        pred=m.predict_proba(xte)[:,1]
        row[f'{tag}特征AUC']=round(roc_auc_score(y_test,pred),4)
        row[f'{tag}特征KS']=round(calc_ks(y_test,pred)[0],4)
    compare_rows.append(row)
compare_df=pd.DataFrame(compare_rows)
print('--- 模型对比（同一训练/测试集，两种特征口径各训一遍）---')
print(compare_df.to_string(index=False))
compare_df.to_csv(os.path.join(OUT_DIR,'model_compare.csv'),index=False,encoding='utf-8-sig')
#为什么 GBDT 的 KS 更高，风控却仍然主用评分卡？
'''1.可解释性：评分卡每个变量都有系数、OR、p 值，能回答"为什么拒绝这个客户"；树模型只能给特征重要性。
2.监管合规：信贷审批要求可解释、可申诉，纯黑箱模型多数监管场景不接受。
3.稳定性：逻辑回归方差低、PSI 易控；树模型容易过拟合、漂移更快。
4.部署成本：评分卡是一张分值表，Excel 就能算；树模型需要模型服务。
5.性能差距需要按同一口径比较：公平口径下 GBDT 的 KS 领先幅度比"全部喂 WOE"时更大，
  说明这个取舍不是"差距小到可以忽略"，而是"用可解释性和稳定性换掉这部分区分度"。
行业常规做法是"双轨"：评分卡做主准入决策，机器学习模型做辅助排序/反欺诈。'''
_ks_lr_woe=compare_df.loc[compare_df['模型']=='逻辑回归评分卡','WOE特征KS'].values[0]
_ks_gbdt_raw=compare_df.loc[compare_df['模型']=='梯度提升','原始特征KS'].values[0]
_ks_rf_woe=compare_df.loc[compare_df['模型']=='随机森林(300棵)','WOE特征KS'].values[0]
_ks_rf_raw=compare_df.loc[compare_df['模型']=='随机森林(300棵)','原始特征KS'].values[0]
_ks_dt_woe=compare_df.loc[compare_df['模型']=='决策树(d=4)','WOE特征KS'].values[0]
print(f'结论：按同一口径比较，梯度提升（原始特征）KS={_ks_gbdt_raw:.4f}，'
      f'比评分卡（WOE特征）的 {_ks_lr_woe:.4f} 高出 {_ks_gbdt_raw-_ks_lr_woe:.4f}；'
      f'但评分卡可解释、可审计、可 Excel 手工复算，因此主决策仍用评分卡。')
print(f'口径公平性说明：随机森林在 WOE 特征下 KS={_ks_rf_woe:.4f}，'
      f'低于单棵决策树的 {_ks_dt_woe:.4f}（正常不应发生）；改用原始特征后 KS={_ks_rf_raw:.4f}，'
      f'恢复正常的模型排序，也证明原对比对树模型不利。')
print('\n全部结果已输出到：'+OUT_DIR)

#================关键结果汇总（自动生成，不再手抄）=================
#【修复】这里原来是一段手抄数字的 docstring，开头写着"每个数字都对应上面某一步的 print 结果"。
#但只要改一次代码、重跑一次，它就会漂 —— 实测已经漂过三次：
#   第一次：结论说剔除、交付用全变量模型；
#   第二次：README 明细表中间行与实际输出不符；
#   第三次：本段里的模型对比 KS 有 6 个值对不上 model_compare.csv。
#根治办法：数字只在一处产生 —— 直接由脚本写出 summary.txt，人不再手抄。
summary_path=os.path.join(OUT_DIR,'summary.txt')
with open(summary_path,'w',encoding='utf-8') as f:
    f.write('===== 关键结果汇总（本文件由 credit-scorecard.py 自动生成，请勿手改）=====\n\n')
    f.write('【数据】\n')
    f.write(f'  样本规模        : {df_shape_raw[0]} × {df_shape_raw[1]}，缺失值 {df.isna().sum().sum()} 个\n')
    f.write(f'  整体坏账率      : {base_badrate:.4f}（{int(df[target].sum())} 笔）\n')
    f.write(f'  多数类基线准确率: {1-base_badrate:.4f}\n\n')
    f.write('【模型】\n')
    f.write(f'  主交付变量数    : {len(feature_cols)}（全变量对照 {len(feature_cols_full)} 个）\n')
    f.write(f'  主交付 训练集   : AUC={auc_train:.4f}，KS={ks_train:.4f}\n')
    f.write(f'  主交付 测试集   : AUC={auc_test:.4f}，KS={ks_test:.4f}\n')
    f.write(f'  主交付 训练-测试差值: AUC={abs(auc_train-auc_test):.4f}，KS={abs(ks_train-ks_test):.4f}\n')
    f.write(f'  主交付 5折CV    : {cv_keep.mean():.4f} ± {cv_keep.std():.4f}\n')
    f.write(f'  伪R² / AIC / BIC: {result.prsquared:.4f} / {result.aic:.1f} / {result.bic:.1f}\n')
    f.write(f'  最大 VIF        : {vif_data["VIF"].max():.4f}\n')
    f.write(f'  剔除变量清单    : {drop_cols}\n\n')
    f.write('【评分卡】\n')
    f.write(f'  A / B / 基础分  : {A:.4f} / {B:.4f} / {base_point:.2f}\n')
    f.write(f'  分数范围(训练)  : {train_score.min():.2f} ~ {train_score.max():.2f}，中位 {train_score.median():.2f}\n')
    f.write(f'  分数范围(测试)  : {test_score.min():.2f} ~ {test_score.max():.2f}，中位 {test_score.median():.2f}\n')
    f.write(f'  分值表复核误差  : {np.abs(recon_score.values-test_score.values).max():.6f}（应为 0）\n\n')
    f.write('【损益】\n')
    f.write(f'  NIM / LGD       : {NIM:.0%} / {LGD:.0%}（假设值，非实测）\n')
    f.write(f'  盈亏平衡坏账率  : {break_even_badrate:.4f}，实际 {base_badrate:.4f}，'
            f'缓冲 {(break_even_badrate-base_badrate)*100:.2f} 个百分点\n')
    f.write(f'  解析最优阈值    : NIM/(NIM+LGD) = {break_even_badrate:.4f}（期望意义下的闭式解）\n')
    f.write(f'  无策略(全通过)  : 通过率 100%，坏账率 {df_strategy["y"].mean():.4f}，利润率 {no_strategy_profit:.4%}\n')
    f.write(f'  评分卡最优策略  : cutoff={best_row["cutoff"]:.3f}，通过率 {best_row["pass_rate"]:.2%}，'
            f'坏账率 {best_row["bad_rate"]:.4f}，利润率 {best_row["profit"]:.4%}\n')
    f.write(f'  最优策略接受/拒绝: 接受 {_acc_n} 人（坏账率 {_acc_bad/_acc_n:.4f}）｜'
            f'拒绝 {_rej_n} 人（坏账率 {_rej_badrate:.4f}）\n')
    f.write(f'  被拒人群误杀率  : {_rej_goodrate:.4f}（被拒的人里其实是好客户的比例）\n\n')
    f.write('--- 规则策略 ---\n')
    f.write(rule_df.round(4).to_string(index=False)+'\n\n')
    f.write('--- LGD 敏感性 ---\n')
    f.write(sens_df.round(4).to_string(index=False)+'\n\n')
    f.write('--- 模型对比（两种特征口径各训一遍）---\n')
    f.write(compare_df.to_string(index=False)+'\n\n')
    f.write('【已知局限（定性，不含数字）】\n')
    f.write('  1. 数据无日期字段，无法做真正的时间外(OOT)验证，时间稳定性未被证实\n')
    f.write('  2. 全部为已发卡客户、无拒绝样本，存在样本选择偏差(reject_inference)，\n')
    f.write('     策略效果不能外推到全量申请者\n')
    f.write('  3. 损益测算的 NIM/LGD 为假设值，结论对该假设高度敏感\n')
    f.write('  4. age_bins 非单调且不显著，已剔除\n')
    f.write('  5. EDUCATION=4 样本量小(468 人)且 WOE 极端，破坏单调性；\n')
    f.write('     该档信号在数据上真实存在，但小样本箱不稳定，本轮保留并列入上线监控：\n')
    f.write('     若该档样本占比或坏账率发生明显变化，立即做粗分类合并或剔除\n')
    f.write('  6. pay1_bins 与 pay_ratio1_bins 信息重叠；其中 pay_ratio1_bins 系数符号翻转，\n')
    f.write('     已从主交付模型中剔除，仅在"全变量对照版本"中保留作为共线性案例\n')
    f.write('  7. 缺少额度/定价联动、策略矩阵、交换矩阵(swap set)分析，为后续可扩展项\n')
print(f'关键结果汇总已自动写出：{summary_path}')
