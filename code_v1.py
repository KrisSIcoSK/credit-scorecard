import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 【新增】全局字体配置，解决中文不显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题
df=pd.read_excel(r'E:\统计学习\数据集\信贷项目\default of credit card clients.xls',header=1)
# 设置显示所有列
pd.set_option('display.max_columns', None)
# 设置显示所有行
pd.set_option('display.max_rows', None)
# 设置列的最大宽度（防止内容被截断）
pd.set_option('display.max_colwidth', None)
# 设置显示宽度（适应更宽的表格）
pd.set_option('display.width', None)

#================数据加载与清洗=================
'''
#确认形状、类型、缺失值
print(df.shape)
print(df.info())
print(df.describe(include='all'))
'''
#df=df.drop('ID',axis=1)#ID列纯标识符，无泛化信息,后续训练模型再删
#EDUCATION（教育程度）1=研究生、2=大学、3=高中、4=其他
#MARRIAGE（婚姻状况） 1=已婚、2=单身、3=其他
#pay_1到pay_6记录客户过去 7 个月每月的还款状态:-2表示没有借款，-1表示已缴清，0表示当月有余额，但按时还了最低还款额，正数表示逾期的月份
#BILL_AMT1 ~ BILL_AMT6表示上月账单金额（应还的账单总额）
#PAY_AMT1 ~ PAY_AMT6表示上月已偿还金额（实际还了多少）
df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4})
df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
#print(df['EDUCATION'].value_counts())
#print(df['MARRIAGE'].value_counts())
#BILL_AMT会出现负值，属于溢缴款、退款、费用返还等是正常现象
df.rename(columns={'PAY_0':'PAY_1'},inplace=True)#将pay_0改成pay_1
pay_atm_cols=[f'PAY_AMT{i}'for i in range(1,7)]
all_zero=(df[pay_atm_cols]==0).all(axis=1)
#print(df[all_zero])
#pay_1到pay_6不构成数据泄露，都是历史逐月逾期状态而非预测时刻之后的数据
#计算基线标准，后续训练模型效果必须大于77.88%，模型才有价值
#print(df['default payment next month'].value_counts(normalize=True))


#================单变量 EDA 与坏账率分析=================
#算 SEX / EDUCATION / MARRIAGE 的分组坏账率,可以直观看到不同性别，学历，婚姻人群的违约差异
target='default payment next month'
def badrate_analysis(df,col):
    res=df.groupby(col)[target].agg(['count','mean'])
    return res
#print(badrate_analysis(df,'SEX'))
#print(badrate_analysis(df,'EDUCATION'))
#print(badrate_analysis(df,'MARRIAGE'))
#算 PAY_1 每个取值的坏账率，画出趋势
pay1_bad=badrate_analysis(df,'PAY_1')
#print(pay1_bad)
plt.plot(pay1_bad.index,pay1_bad['mean'],marker='o')
plt.xlabel('pay_1的逾期状态')
plt.ylabel('坏账率')
plt.title('pay_1各取值坏账率趋势')
#plt.show()
plt.close()
#把 PAY_0 按 >=1 / <=0 二分，对比两组坏账率
df['pay1_bin']=df['PAY_1'].apply(lambda x:'逾期(>=1)'if x>=1 else'正常(<=0)')
pay1_bin_bad=df.groupby('pay1_bin')[target].agg(['count','mean'])
#print(pay1_bin_bad)#逾期客户坏账率远高于无逾期客户
#对 LIMIT_BAL 做等频分箱，看坏账率是否单调
df['limit_bal_bins']=pd.qcut(df['LIMIT_BAL'],q=10)
#print(df.groupby('limit_bal_bins')[target].agg(['count','mean']))#可以看到坏账率随着授信金额增大而减少
#对 AGE 做等频分箱，观察分布形状
df['age_bins']=pd.qcut(df['AGE'],q=10)
#print(df.groupby('age_bins')[target].agg(['count','mean']))#年龄坏账率偏U型，年轻人和老年人坏账偏高，中华人最低
#对 BILL_AMT1 / PAY_AMT1 做等频分箱
for i in range(1,7):
  df[f'bill{i}_bins']=pd.qcut(df[f'BILL_AMT{i}'],q=5)
#print(df.groupby('bill1_bins')[target].agg(['count','mean']))#bill分箱之后几乎无规律
for i in range(1,7):
  df[f'pay{i}_bins']=pd.qcut(df[f'PAY_AMT{i}'],q=5,duplicates='drop')
#print(df.groupby('pay1_bins')[target].agg(['count','mean']))#pay分箱之后完美单调递减
#构造衍生特征 额度使用率 = BILL_AMT1 / LIMIT_BAL 并分析
df['util_use']=df['BILL_AMT1']/(df['LIMIT_BAL']+1e-6)
df['util_use_bins']=pd.qcut(df['util_use'],q=10)
#print(df.groupby('util_use_bins')[target].agg(['count','mean']))#整体递增（使用率越高风险越高，符合业务逻辑）
#构造衍生变量delay_cnt(逾期次数)
pay_cols=[f'PAY_{i}'for i in range(1,7)]
df['delay_cnt']=((df[pay_cols]>0).sum(axis=1))
df['delay_cnt_bins']=pd.qcut(df['delay_cnt'],q=10,duplicates='drop')
#print(df.groupby('delay_cnt_bins')[target].agg(['count','mean']))
#构造衍生变量pay_ratio1(还款比例)
df['BILL_AMT1_safe']=df['BILL_AMT1'].replace(0,1)
df['pay_ratio1']=(df['PAY_AMT1']/df['BILL_AMT1_safe'])
df['pay_ratio1_bins']=pd.qcut(df['pay_ratio1'],q=5,duplicates='drop')
#print(df.groupby('pay_ratio1_bins')[target].agg(['count','mean']))
#检验 ID 能否作为时间代理（做卡方检验）
df['id_bins']=pd.qcut(df['ID'],q=5)
cross_table=pd.crosstab(df['id_bins'],df[target])
ch2,p,dof,expected=chi2_contingency(cross_table)
#print(p)#p值很小拒绝原假设，但各段坏账率极差只有 6.6 个百分点且无单调趋势，考虑 N=30000 时卡方检验极度敏感，我判断这只是大样本下的统计显著而非真实的时间漂移，因此不采用 ID 作为时间代理
#业务发现：1.还款历史是最强信号：有过逾期（PAY_0≥1）的客户坏账率 50.29%，是无逾期客户（13.83%）的 3.6 倍，而这类客户占 22.73%
#2.额度越低风险越高：授信额度 ≤3 万的客户坏账率 35.85%，是 >36 万客户（11.87%）的 3 倍
#3.还款金额是保护因素：月还款 ≤316 元的客户坏账率 34.22%，是还款 >10300 元客户（12.78%）的 2.7 倍
#4.年龄呈 U 型：25 岁以下和 49 岁以上风险偏高，中间段最稳
#5.账单金额几乎无预测力：BILL_AMT1 各分箱坏账率在 19.67%~25.53% 之间无规律波动 —— 反直觉但重要：欠多少钱不如"还得怎么样"重要


#================分箱与单调性=================
#为 PAY_0~PAY_6 设计自定义粗分类，解决不单调 + 小样本箱(高逾期数字样本极少会出现小样本箱)
def bin_pay(x):
    if x<=-1:
        return'按时结清'
    elif x==0:
        return'循环账户'
    elif x==1:
        return'逾期1个月'
    elif x>=2:
        return '逾期2个月以上'
for i in range(1,7):
    df[f'PAY{i}_bin'] = df[f'PAY_{i}'].apply(bin_pay)
#print(df.groupby('PAY1_bin')[target].agg(['count','mean']))#人工分箱且坏账率单调
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
#print(df.groupby('limit_bin_final')[target].agg(['count','mean']))#LIMIT_BAL人工分箱后单调
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
#print(df.groupby('util_bins')[target].agg(['count','mean']))#额度使用率通过人工分箱之后变成单调递增
bins_break = [-np.inf, 0, 0.2, 0.5, 1, np.inf]
bin_labels = [
     "负数(溢缴款)",
     "0（完全不还款）",
     "(0,0.2] 少量还款",
     "(0.2,0.5] 部分还款",
     "≥0.5 大额/全额还款"
 ]
df['pay_ratio1_bins'] = pd.cut(df['pay_ratio1'],bins=bins_break,labels=bin_labels,include_lowest=True)
#print(df.groupby('pay_ratio1_bins')[target].agg(count = 'count',bad_rate = 'mean').reset_index())
#检查每一箱的样本量占比，低于 5% 的箱必须合并
def check_bin_ratio(df,bin_col):
    ratio=df[bin_col].value_counts(normalize=True).sort_values(ascending=False)
    return(ratio)
#print(check_bin_ratio(df,'limit_bin_final'))
#print(check_bin_ratio(df,'pay1_bins'))
#print(check_bin_ratio(df,'util_bins'))#检查所有分箱样本数都不小于百分之5
#合并后重新计算各箱坏账率，验证单调性(已验证)
'''思考：箱数多好还是少好？说出权衡 
箱数多：捕捉更细的风险差异，IV更高；缺点：容易出现小样本箱，分箱切点不稳定，容易过拟合，PSI漂移风险高
箱数少：样本充足、稳定性强、业务好解释；缺点：风险区分变粗糙，IV下降，丢失细节信息
实操折中：一般4~6箱。在保证样本充足+单调前提下，尽可能多保留箱数。'''
#检查分箱的稳定性：随机抽两半数据分别分箱，切点是否一致...


#================WOE / IV 计算与变量筛选=================
PAY1_bad=df.groupby('PAY1_bin')[target].agg(['count','mean'])
limit_bin_bad=df.groupby('limit_bin_final')[target].agg(['count','mean'])
pay1_atm_bad=df.groupby('pay1_bins')[target].agg(['count','mean'])
util_bins_bad=df.groupby('util_bins')[target].agg(['count','mean'])
#这四个是用来看分箱结果以及单调性的
#定义WOE函数
def calc_woe(df,bin_col,target_col):
    gp=df.groupby(bin_col)[target_col].agg(bad='sum',total='count').reset_index()
    gp['good']=gp['total']-gp['bad']
    total_bad=gp['bad'].sum()
    total_good=gp['good'].sum()
    gp['good_s'] = gp['good'] + 1
    gp['bad_s'] = gp['bad'] + 1
    total_bad_s = total_bad + len(gp)
    total_good_s = total_good + len(gp)
    gp['bad_pct']=(gp['bad_s'])/total_bad_s
    gp['good_pct']=(gp['good_s'])/total_good_s
    gp['WOE']=np.log(gp['bad_pct']/gp['good_pct'])
    gp.rename(columns={bin_col: "分箱区间"}, inplace=True)
    gp["变量名"] = bin_col
    gp = gp[["变量名", "分箱区间", "bad", "total", "good", "bad_pct", "good_pct", "WOE"]]
    return gp
'''
print(calc_woe(df,'PAY1_bin',target))
print(calc_woe(df,'limit_bin_final',target))
print(calc_woe(df,'pay1_bins',target))
print(calc_woe(df,'util_bins',target))
#计算四个WOE值'''
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
#print(iv_df)#求解26个变量的IV
'''解释为什么 IV>0.5 要警惕:
1.IV过高代表变量区分力太强，大概率标签泄漏（变量里直接包含了未来违约信息）
2.现实业务里信贷特征很难做到IV>0.5；往往是目标信息混入特征
3.极易过拟合，OOT样本上AUC暴跌，上线漂移严重'''
#print(iv_df[iv_df['iv']>0.02])#筛选变量清单
#规则：保留 IV ≥0.02；IV<0.02剔除。
#剔除理由：对违约几乎没有区分能力，噪声大，入模会干扰模型、增加不稳定。
#构造一个衍生特征，观察它的 IV 是否"虚高":构造的delay_cnt就是完全由pay系列原始变量计算得到属于信息重复
#而额度使用率/ PAY_RATIO1是两个不同维度字段相除，产生新业务维度，不是虚高
#输出最终 WOE 映射表
woe_map_all=pd.DataFrame()
for feat in col_list:
    woe_df=calc_woe(df,feat,target)
    woe_map_all=pd.concat([woe_map_all,woe_df],ignore_index=True)
#print(woe_map_all)
#woe_map_all.to_csv(r"E:\统计学习\数据集\信贷项目\woe映射表.csv",index=False,encoding='utf-8-sig')
#对比：筛掉 BILL_AMT 六个变量损失了多少 IV...


#================共线性诊断=================
#算所有数值变量的相关矩阵，找出 |r|>0.7 的组合
num_cols = ["LIMIT_BAL","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
             "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
             "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",'util_use','delay_cnt','pay_ratio1']
corr_matrix=df[num_cols].corr()
#print(corr_matrix)#计算所有系数的相关系数矩阵
corr_pairs=[]
for i in range(len(corr_matrix.columns)):
    for j in range(i+1,len(corr_matrix.columns)):
        r=corr_matrix.iloc[i,j]
        if abs(r) > 0.7:
          corr_pairs.append((corr_matrix.columns[i],corr_matrix.columns[j],r))
res_df=pd.DataFrame(corr_pairs,columns=['变量1','变量2','相关系数r'])
#print(res_df)#找出相关系数大于0.7的组合
#解释 BILL_AMT1~6 之间 r>0.8 的含义和后果
'''相关系数 r>0.8，说明连续6个月账单金额高度正相关。上个月账单高的客户，下个月账单大概率也高；6个账单变量承载大量重复信息。
后果：1.6个变量信息大量重叠，放入逻辑回归会出现多重共线性；
2.回归系数估计不稳定，系数正负号可能反常，p值不可靠；
3.VIF膨胀；模型很难判断到底是哪一期账单在影响违约；
4.业务冗余，6个变量干的几乎是一件事，不需要全部入模。'''
#检查 PAY_* 变量之间的相关性
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
#按 7:3 分层切分训练/测试集
# y：违约标签
y = df["default payment next month"]
# X：所有预测特征
feature_cols =['limit_bin_final','EDUCATION','age_bins',*[f'PAY{i}_bin' for i in range(1,7)],
    *[f'pay{i}_bins' for i in range(1,7)],'util_bins','pay_ratio1_bins']
X=df[feature_cols]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
#用训练集拟合 WOE 映射，再应用到测试集
df_test=x_test.copy()
df_train=x_train.copy()
df_train['default']=y_train
woe_mapping_dict={}
for feat in feature_cols:
    woe_table=calc_woe(df_train,feat,'default')
    #print(woe_table)
    temp_map = dict(zip(woe_table['分箱区间'], woe_table['WOE']))
    woe_mapping_dict[feat]=temp_map#初始化一个空字典，用来存所有特征的 WOE 映射关系。
x_train_woe=x_train.copy()
for feat in feature_cols:
    x_train_woe[feat]=x_train[feat].map(woe_mapping_dict[feat])#把训练集每个特征的分箱标签替换成对应的 WOE 数值
x_test_woe=x_test.copy()
for feat in feature_cols:
    x_test_woe[feat]=x_test_woe[feat].map(woe_mapping_dict[feat])#用训练集拟合出的同一套映射替换测试集，避免数据泄露
#先建基线模型：用训练集均值预测
base_prob=y_train.mean()
y_pred_base=np.full(len(y_test),fill_value=base_prob)
base_auc=roc_auc_score(y_test,y_pred_base)
#print(f'基线模型测试集AUC={base_auc:.4f}')
#训练 LogisticRegression，算 AUC
lr=LogisticRegression(random_state=42,max_iter=1000)
lr.fit(x_train_woe,y_train)
#预测概率，取正类(违约)的概率
y_train_pred=lr.predict_proba(x_train_woe)[:,1]
y_test_pred=lr.predict_proba(x_test_woe)[:,1]
auc_train=roc_auc_score(y_train,y_train_pred)
auc_test=roc_auc_score(y_test,y_test_pred)
#print(f'训练集AUC={auc_train:.4f},测试集AUC={auc_test:.4f}')
#KS函数
def calc_ks(y_true,pred_prob):
    df_ks=pd.DataFrame({'y':y_true,'prob':pred_prob})
    df_ks=df_ks.sort_values('prob',ascending=False).reset_index(drop=True)
    df_ks['good']=1-df_ks['y']
    df_ks['bad']=df_ks['y']
    df_ks['cum_good']=df_ks['good'].cumsum()/df_ks['good'].sum()
    df_ks['cum_bad']=df_ks['bad'].cumsum()/df_ks['bad'].sum()
    df_ks['diff']=df_ks['cum_bad']-df_ks['cum_good']
    ks_value=df_ks['diff'].max()
    return ks_value,df_ks
ks_train,_=calc_ks(y_train,y_train_pred)
ks_test,_=calc_ks(y_test,y_test_pred)
#print(f'训练KS={ks_train:.4f},测试KS={ks_test:.4f}')
#对比训练集与测试集指标，判断是否过拟合
#print(f'AUC差值{abs(auc_train-auc_test):.4f}')#计算AUC差值
#print(f'KS差值{abs(ks_train-ks_test):.4f}')#计算KS差值
'''训练集AUC=0.7735,测试集AUC=0.7634,风控模型效果不错
训练KS=0.4142,测试KS=0.3990,KS>0.4区分能力强
AUC差值0.0101,KS差值0.0151,几乎没有过拟合，泛化能力强'''
#做 5 折交叉验证，看 AUC 均值与标准差
cv_auc_list=cross_val_score(lr,x_train_woe,y_train,cv=5,scoring='roc_auc')
#print(f'5折AUC均值:{cv_auc_list.mean():.4f},标准差:{cv_auc_list.std():.4f}')
#5折AUC均值:0.7720,与训练集AUC接近，标准差:0.0161,波动很小，说明模型区分能力良好，泛化能力强，模型稳定，无明显过拟合
#画 ROC 曲线，解释 AUC=0.76 的业务含义
fpr,tpr,_=roc_curve(y_test,y_test_pred)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f'AUC={auc_test:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR假阳性率')
plt.ylabel('TPR真阳性率')
plt.title('ROC曲线')
plt.legend()
#plt.show()
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
#plt.show()
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
#plt.show()
plt.close()


#================statsmodels 系数表与显著性=================
x_train_sm=sm.add_constant(x_train_woe)
#用 statsmodels 构建逻辑回归模型，并在训练集上完成拟合
logit_model=sm.Logit(y_train,x_train_sm)
result=logit_model.fit(disp=0)
#print(result.summary())
df_coef=pd.DataFrame({
    'coef':result.params,
    'std_err':result.bse,
    'z_value':result.tvalues,
    'p_value':result.pvalues,
    'OR':np.exp(result.params),#优势比
    'lower_95':result.conf_int()[0],
    'upper_95':result.conf_int()[1]
 })
#print(df_coef)
#print(df_coef[df_coef['p_value']>0.05])#p值大于0.05(不显著)的系数表
#PAY_2的iv值在前面是第二高，但是在这里却p值很大，系数几乎等于0，完全不显著
'''原因:1.信息被 PAY_0 吸收了	PAY_0 与 PAY_2 相关 r=0.672。"最近一期逾期"已经包含了"上期逾期"的大部分信息
2.统计上叫"冗余"	在控制 PAY_0 之后，PAY_2 不再提供额外信息
3.单变量 IV 高 ≠ 多变量有用	IV 是单变量指标，它不知道其他变量的存在
IV 负责"海选"，逻辑回归负责"终选"。IV 高只是入场券，不代表能留在模型里。'''
#系数符号检查
'''本文档用 WOE = ln(坏/好) 约定，所以"坏账率越高的箱 WOE 越正"，模型学出来的系数应该为正（WOE 越正 → 分数越低 → 风险越高）
实测所有系数均为正，符合预期。'''
#计算 VIF，判断共线性是否可接受
vif_data=pd.DataFrame()
vif_data['feature']=x_train_sm.columns
vif_data['VIF']=[variance_inflation_factor(x_train_sm.values,i)for i in range(x_train_sm.shape[1])]
#print(vif_data)#VIF全部小于4，无严重共线性
#报告伪 R²、AIC 并解释其含义
#print('McFadden伪R²:',result.prsquared)#伪 R² (McFadden)=0.18,表示相对"只用截距"的模型，对数似然改善多少，0.18属于良好
#print('AIC:',result.aic)#AIC拟合优度 − 复杂度惩罚，用于比较模型，不用于绝对评价
#print('BIC:',result.bic)
#风控模型中只看AUC和KS不看伪R²，逻辑回归的伪 R² 天然偏低，0.2 以上就算很好，用 R² 评价评分卡本身是错的方法论。
#决定最终模型：是否剔除不显著变量？给出理由
#剔除，评分卡要"每个变量都有意义"，风控评分卡的标准做法是保留统计显著 + 业务可解释的变量


#================评分卡刻度转换=================
base_score=600
base_odds=20
PDO=50
B=PDO/np.log(2)
A=base_score - B*np.log(base_odds)
def prob_to_score(p_bad):
    p_bad = np.clip(p_bad, 1e-6, 1-1e-6)
    return A + B*np.log((1-p_bad)/p_bad)   # odds = 好/坏
#print(prob_to_score(1/(1+20)))
#方向自检
p_high_risk=0.30#高违约概率
p_low_risk=0.0476#低违约概率
score_high=prob_to_score(p_high_risk)
score_low=prob_to_score(p_low_risk)
#print(f'高风险用户(p=0.3)分数:{score_high:.2f}')
#print(f'低风险用户(p=0.0476)分数:{score_low:.2f}')
#计算每一个变量的分值表
coef_series=result.params
beta0=coef_series.iloc[0]#const截距项
coef_vars=coef_series.iloc[1:]#各个入模特征的β系数
base_point=A+B*beta0
#print(f'基础分:{base_point:.2f}')
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
    score_card=pd.concat(score_card_list)
    score_card['bin_score']=score_card['bin_score'].round(2)
    #print('评分卡分值表')
    #print(score_card)
logit_train = beta0 + x_train_woe @ coef_vars
train_score = A - B * logit_train
logit_test = beta0 + x_test_woe @ coef_vars
test_score = A - B * logit_test
# 打印描述统计，看分数范围
#print("=====训练集分数统计=====")
#print(train_score.describe())
#print("=====测试集分数统计=====")
#print(test_score.describe())
# 绘制分布对比图
plt.figure(figsize=(9,5))
plt.hist(train_score, bins=30, alpha=0.6, label='Train Score')
plt.hist(test_score, bins=30, alpha=0.6, label='Test Score')
plt.xlabel('评分卡分数')
plt.ylabel('样本数量')
plt.title('7.5 评分卡全样本分数分布')
plt.legend()
plt.grid(alpha=0.3)
#plt.show()
plt.close()
#找一个真实客户，手工算出他的分数...


#================PSI 稳定性监控=================
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
    base_cnt = df_base[bin_col].value_counts(normalize=True).sort_index()
    actual_cnt = df_actual[bin_col].value_counts(normalize=True).sort_index()
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
#print('训练集vs随机切分测试集PSI')
#print(psi_df)
#解释为什么随机切分的 PSI 必然接近 0
'''随机切分：把同一时间段全部样本打乱，随机拆成train/test。
样本是同分布、同一时间、同一批客群，只是随机抽一部分。每个分箱里面样本占比在train和test几乎完全一样。
Actual_i / Expected_i，(Actual-Expected)趋近0，PSI求和结果自然接近0。
随机划分只能检验模型拟合能力，不能检验时间稳定性！ 真实业务漂移是时间带来的，不是随机抽样。'''
#按 ID 分段做"伪 OOT"，看 PSI 变化
df_all=pd.concat([df_train,df_test]).reset_index(drop=True)
split_pos=int(len(df_all)*0.7)
df_pseudo_train=df_all.iloc[:split_pos]#前70%当作开发样本
df_pseudo_oot=df_all.iloc[split_pos:]#后30%当作伪OOT未来样本
#计算伪OOT的PSI
pseudo_psi={}
for feat in feature_cols:
    psi_val=get_feature_psi(df_pseudo_train,df_pseudo_oot,feat)
    pseudo_psi[feat]=psi_val
pseudo_psi_df=pd.DataFrame(list(pseudo_psi.items()),columns=['feature','PSI'])
#print('伪OOT的PSI')
#print(pseudo_psi_df)
#全部 PSI < 0.011，判定"稳定"。——这个"稳定"也不能当作真实的时间稳定性证据。
#本数据集无法做真 OOT 的局限及影响
'''1.无法进行时间外（OOT）验证。数据集不含日期字段，仅有 1~30000 的顺序 ID考虑到 N=30000 时卡方检验极度敏感，
我判断这是大样本下的统计显著而非真实时间漂移，因此不采用 ID 作为时间代理。
2.影响：模型的时间稳定性未经验证。真实业务中，客群结构、宏观环境、竞品策略都会随时间变化，模型可能在 6~12 个月后失效。
3.PSI 检验的局限：随机切分下的 PSI≈0.002 是数学必然，不构成稳定性证据
4.样本选择偏差：本数据全部为已发卡客户，不含拒绝样本，因此策略效果不能外推到全量申请者
5.如果给我真实数据，我会：按放款月份切分做真正的 OOT（训练早期、验证后期），并按月计算 PSI 做漂移监控，用冠军挑战者框架小流量验证策略。'''
#设计一份上线后的监控方案（指标+阈值+动作）...


#================策略设计与损益测算=================
#设定损益参数（NIM、LGD），真实场景NIM取自产品定价、资金成本；LGD取自历史催收回收数据。本数据集没有资金、催收数据，所以为假设参数。
#NIM(净息差):放款的收益率，LGD(违约损失率):客户违约后，每一元本金最终亏掉的比例
NIM=0.20
LGD=0.70
#计算盈亏平衡坏账率
break_even_badrate=NIM/(NIM+LGD)
#print(f'盈亏平衡坏账率{break_even_badrate:.2%}')
#计算无策略时的总损益与利润率(基线)
y_pred_prob=result.predict(x_train_sm)
y_true=y_test.reset_index(drop=True)
df_strategy=pd.DataFrame({'prob':y_pred_prob,'y':y_true})
#遍历不同cutoff，计算每一个门槛下：通过率、坏账率、净利润
cutoff_list = np.linspace(0,0.6,30)
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
#print(res_df)
#画通过率-坏账率-损益曲线，找最优点
plt.figure(figsize=(10,6))
plt.plot(res_df['pass_rate'],res_df['bad_rate'],label='坏账率')
plt.plot(res_df['pass_rate'],res_df['profit'],label='净利润')
plt.xlabel('通过率')
plt.legend()
plt.title('通过率-坏账率-利润曲线')
#plt.show()
plt.close()
#找出利润最大的最优cutoff
beat_row=res_df.loc[res_df['profit'].idxmax()]
#print('最优策略点')
#print(beat_row)
#分析通过率-坏账率-损益曲线
'''1.通过率上升时，暴涨，被拒的是评分最低的高危客户，他们的预期损失远大于收益。拒掉他们是"纯赚"
2.继续上升但边际递减，越往后拒的人，风险越低，"避免的损失"和"放弃的收益"逐渐接近
3.坏账率峰值，此时边际收益 = 边际损失
4.之后下降，被拒的客户已经是好客户为主了。拒掉他们放弃的收益大于避免的损失 → 亏'''
#风控的目标不是"把坏账率降到最低"，而是"让风险调整后的利润最大"。
#设计 3~5 条业务规则，算命中率/命中坏账率/边际收益
'''# 9.6 硬规则测算，注意替换df_test内真实字段名
df_rules = df_test.copy()
df_rules['y'] = y_test.reset_index(drop=True)

# 自定义3条风控硬规则，修改字段为你数据集真实列
rule_dict = {
    "历史逾期次数≥3": df_rules['PAY1_bin'] >=3,
    "信贷使用率>90%": df_rules['util_bins'] >0.9,
    "近3个月查询次数>6次": df_rules['query_bin']>6
}
rule_result = []

for rule_name, mask in rule_dict.items():
    hit_cnt = mask.sum()
    hit_rate = hit_cnt / len(df_rules)
    if hit_cnt > 0:
        hit_badrate = df_rules.loc[mask, "y"].mean()
    else:
        hit_badrate = 0
    # 边际收益
    marginal_profit = hit_cnt * hit_badrate * LGD - hit_cnt * (1-hit_badrate)*NIM
    rule_result.append({
        "规则名称": rule_name,
        "命中人数": hit_cnt,
        "命中率": hit_rate,
        "命中坏账率": hit_badrate,
        "边际收益": marginal_profit
    })

rule_df = pd.DataFrame(rule_result)
print("硬规则评估结果")
print(rule_df.to_string(index=False))'''
#对比"评分卡策略" vs "单条规则策略"
'''硬规则策略：一刀切，满足条件直接拒绝，简单易解释，但风险分层粗糙，容易误杀优质客户。
评分卡策略：连续风险分数精细化分层，可以灵活调整cutoff，最大化利润；缺点是建模复杂，业务理解成本更高。行业常用组合方案：硬规则前置拦截极端风险，评分卡做主准入策略。'''
#做敏感性分析：LGD代表违约损失，LGD越高，单个坏客户造成亏损越大。LGD上升时，最优cutoff会抬高，准入策略收紧；LGD降低，可以适当放宽准入，接受更高坏账。
'''df_strategy = pd.DataFrame({
    'prob': result.predict(df_test),
    'y': y_test.reset_index(drop=True)
}).reset_index(drop=True)
# 定义总样本数
total_num = len(df_strategy)
lgd_list = [0.5, 0.6, 0.7, 0.8]
for lgd_val in lgd_list:
    profit_array = []
    for cutoff in cutoff_list:
        accept_mask = df_strategy['prob'] < cutoff
        accept_count = accept_mask.sum()
        pass_rate = accept_count / total_num
        if accept_count == 0:
            profit = 0
        else:
            accept_df = df_strategy[accept_mask]
            bad_rate = accept_df['y'].mean()
            loan = pass_rate * total_num
            interest = loan*(1-bad_rate)*NIM
            loss = loan*bad_rate*lgd_val
            profit = interest - loss
        profit_array.append(profit)
    best_cut = cutoff_list[np.argmax(profit_array)]
    print(f"LGD={lgd_val}, 最优cutoff(违约概率阈值) = {best_cut:.3f}")'''
#写"策略上线方案"（灰度、冠军挑战者、监控）
'''1.离线验证	本报告（OOT 验证 + 损益测算）	利润率提升且通过率不低于业务底线
2.灰度上线	5% 流量先跑新策略，与旧策略并存	观察 1~2 个账期，实际坏账率与预测偏差 <20%
3.冠军挑战者	新旧策略各 10% 流量对比	新策略利润率显著优于旧策略
4.全量	逐步放量至 100%
5.持续监控	监控表'''
#用业务语言总结：这个策略值不值得上...


#================模型对比与结论=================