import pandas as pd
import os
from scipy import stats
from statsmodels.stats import anova
import statsmodels.api as sm
from statsmodels.formula.api import ols

wd = os.path.abspath(os.getcwd())

mydat = pd.read_csv(wd + '/Placement_Data_Full_Class.csv')
print(type(mydat))

# 維度
print(mydat.shape)
print(len(mydat))
cols = (mydat.columns)
print(cols.tolist())


# types of columns
import numpy as np

is_numeric = mydat.dtypes != 'object'
numeric_cols = mydat.columns[is_numeric]


# missing values
missing_cols = mydat.columns[np.where(mydat.isna())[1]]
print(np.unique(missing_cols))
print(missing_cols.value_counts() / len(mydat))
### salary missing deeper explore
not_placed = np.where(mydat['status']=='Not Placed')[0]
salary_missing = np.where(mydat['salary'].isna())[0]

print((not_placed == salary_missing).all())



# numeric distribution
## distribution
pd.set_option("display.max_columns", 15)
print(mydat.describe(include=[np.number]))

## outlier detect
def inRangeCheck(x, left, right): # x: series, left: int, right: int; outlier: list
    #outlier = [x[i] for i in range(len(x)) if x[i]<left] + [x[i] for i in range(len(x)) if x[i]>right]
    outlier = pd.concat([x[x<left], x[x>right]]).sort_values()
    return outlier

def outlierDetect(x): # x: series; outlier: list
    x_series = pd.Series(data=x)
    q1 = x_series.quantile(.25)
    q3 = x_series.quantile(.75)
    iqr = q3 - q1
    outlier = pd.concat([x_series[x_series<(q1-1.5*iqr)], x_series[x_series>(q1+1.5*iqr)]]).sort_values()
    return outlier


cols = [nc for nc in numeric_cols if nc not in ['sl_no','salary']]
out_range = mydat.loc[:,cols].apply(inRangeCheck, left=0, right=100)
out_salary = inRangeCheck(mydat.loc[:,'salary'], left=0,right=float('inf'))
print('Out of range: ', out_range, '\n')
print('Negative salary: ', out_salary, '\n')

cols = [nc for nc in numeric_cols if nc not in ['sl_no']]
outlier_normal = []
for col in cols:
    temp = outlierDetect(mydat.loc[:, col])
    outlier_normal.append(temp)

outlier_len = {}
for i in range(len(outlier_normal)):
    outlier_len.update({outlier_normal[i].name :len(outlier_normal[i])})

print("outlier_len: ", outlier_len, '\n\n')

## correlation
corr = mydat.loc[:,numeric_cols].corrwith(mydat['salary'])
print(corr)


# categorical distribution
categorical_cols = mydat.columns[mydat.dtypes == 'object']
print(type(categorical_cols))

frequency = {}
i = 0
for cc in categorical_cols:
    if i == 0:
        print('Count Values: \n')
    temp = mydat[cc].value_counts()
    frequency.update({cc: temp})
    print(temp)
    i += 1



# Visualization
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Single variable distribution check
### Numeric Variables
plot_numeric_cols = [nc for nc in numeric_cols if nc not in ['sl_no']]
n_row = 3
n_col = 2
fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(8,7))
plt.subplots_adjust(hspace=0.7)
count=0
for i in range(n_row):
    for j in range(n_col):
        sns.distplot(mydat[plot_numeric_cols[count]],
            ax=axes[i,j])
        axes[i,j].set_title(plot_numeric_cols[count], fontsize=15)
        count+=1

#plt.savefig(wd + '/numeric_dist.png')
#plt.show()

#### box plot
fig, axes = plt.subplots(nrows=n_row, ncols=n_col)
plt.subplots_adjust(hspace=0.7)

count=0
for i in range(n_row):
    for j in range(n_col):
        sns.boxplot(mydat[plot_numeric_cols[count]],
            ax=axes[i,j])
        axes[i,j].set_title(plot_numeric_cols[count], fontsize=15)
        count+=1

#plt.show()


### Categorical Variables
n_row = 4
n_col = 2
fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10,7))
plt.subplots_adjust(hspace=0.7, wspace=0.3)
count=0
for i in range(n_row):
    for j in range(n_col):
            sns.barplot(x=frequency[categorical_cols[count]].values, y=frequency[categorical_cols[count]].index,
                ax=axes[i,j])
            axes[i,j].set_title(categorical_cols[count], fontsize=15)
            count+=1

#plt.savefig(wd + '/categorical_eda1.png')


## Which variable is associated with the "status" variable
### Numeric variables
plot_numeric_cols = [nc for nc in numeric_cols if nc not in ['sl_no']]

fig, axes = plt.subplots(nrows=len(plot_numeric_cols), ncols=1, figsize=(7,10))
for i in range(len(plot_numeric_cols)):
    if(plot_numeric_cols[i]!='salary'):
        sns.violinplot(data=mydat, x=plot_numeric_cols[i], y='status', #hue='status',
            cut=0, order=['Placed','Not Placed'], scale='count', bw=.3, orient='h',
            ax=axes[i]) # 指定畫在哪個subplots
    else:
        sns.violinplot(data=mydat, x=plot_numeric_cols[i], y='status', hue='status',
            cut=0, order=['Placed','Not Placed'], scale='count', bw=.3, orient='h',
            ax=axes[i])
    axes[i].set_ylabel(plot_numeric_cols[i], rotation=0, fontsize=15, labelpad=27) # ax.set_ylabel
    axes[i].set_yticks(ticks=[])
    axes[i].set_xlabel('')

axes[0].set_title('EDA of Numeric Variables', fontsize=20)

#plt.savefig(wd + '/Junior_Python/numeric_eda.png')

### catgorical variables
plot_categorical_cols = categorical_cols[categorical_cols != 'status']
#cross_dat = pd.crosstab(index=mydat['status'], columns=[mydat['workex']])

n_row = 4
n_col = 2
fix, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(10,7))
plt.subplots_adjust(hspace=0.7, wspace=0.3)
for i in range(n_row):
    end = False
    for j in range(n_col):
        ind = i*2+j
        if ind>(len(plot_categorical_cols)-1):
            end = True
            break
        col = plot_categorical_cols[ind]
        count_table = mydat.groupby(['status',col]).size().reset_index(name='counts')
        count_table['total'] = count_table['counts'].groupby(count_table[col]).transform('sum')
        count_table['proportion'] = (count_table['counts']/count_table['total']*100).round(2)

        sns.barplot(data=count_table, x=col, y="proportion", hue="status", ax=axes[i,j])
        ys = count_table['proportion'].groupby(count_table[col]).max()
        values = count_table.loc[:,[col,'total']].drop_duplicates(subset=col).set_index(col)['total']

        for x, y, value in zip(range(len(ys)), ys.values, values.values):
            axes[i,j].text(x=x, y=y, s=str(value), fontsize=12, horizontalalignment='center')

        axes[i,j].set_title(col, fontsize=14)
        axes[i,j].set_ylabel('')
        axes[i,j].set_xlabel('')

    if end:
        break
#plt.show()


# Test of different mean of every numeric variable
# #Before hypothesis test we first do data transfomation and normality test
# #salary, etest_p seem to be more likely to be transform to normality
# #Seems etest_p is a little right skrew

#lm_model = sm.ols('etest_p~status', data=mydat).fit()
#print(pd.Series(lm_model.fittedvalues).unique())
#sns.distplot(lm_model.resid)
#plt.show()

mydat['etest_p_trans0'] = mydat['etest_p'] ** .1
#lm_model = sm.ols('etest_p_trans0~status', data=mydat).fit()
#sns.distplot(lm_model.resid)
#plt.show()

mydat['etest_p_trans'] = (mydat['etest_p_trans0']-min(mydat['etest_p_trans0']))*100/(max(mydat['etest_p_trans0'])-min(mydat['etest_p_trans0']))

test_numeric_cols = plot_numeric_cols
test_numeric_cols = ['etest_p_trans' if col == 'etest_p' else col for col in test_numeric_cols]
test_numeric_cols.remove('salary')

t_test = pd.DataFrame(columns=['coef','se','tvalue','pvalue'])
for col in test_numeric_cols:
    X=np.array(mydat['status'].map({'Placed':1, 'Not Placed':0}))
    X = sm.add_constant(X)
    Y = np.array(mydat[col])
    temp = sm.OLS(Y,X).fit()

    df_temp = pd.DataFrame({'coef':[temp.params[1]],
        'se':[temp.bse[1]],
        'tvalue':[temp.tvalues[1]],
        'pvalue':[temp.pvalues[1]]})
    t_test = t_test.append(df_temp, ignore_index=True)

t_test.index = test_numeric_cols
t_test = t_test.apply(lambda x: round(x,2) , axis=0)
print(t_test)

# chi-square independence test
outcome = 'status'
cols = categorical_cols[categorical_cols!='status']
chiind_test = pd.DataFrame(columns=['chi2','pvalue'])
for i in range(len(cols)):
    col = cols[i]
    contingency_table = pd.crosstab(index=mydat[outcome], columns=mydat[col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table, correction=False)
    chiind_test = chiind_test.append(pd.DataFrame({'chi2':[chi2], 'pvalue':[p]}))

chiind_test.index = cols
print(chiind_test)
