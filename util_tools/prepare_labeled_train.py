import pandas as pd
import numpy as np

# ivan file

#df_ivan = pd.read_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/ivan-labels.csv", delimiter="~", names=["tweet_id", "user_id", "unn", "tw_full", "stance"])
#print(df_ivan.shape);
#print(df_ivan.head());

#df_ivan_filtered = df_ivan[pd.notnull(df_ivan['stance'])]
#print(df_ivan_filtered.shape);

#print(df_ivan_filtered.head());
#df_ivan_filtered.stance = df_ivan_filtered.stance.astype(int)

#df_ivan_filtered_remain = df_ivan_filtered[df_ivan_filtered['stance']==0]
#print(df_ivan_filtered_remain.shape);
#print(df_ivan_filtered_remain.head());

#df_ivan_filtered_leave = df_ivan_filtered[df_ivan_filtered['stance']==1]
#print(df_ivan_filtered_leave.shape);
#print(df_ivan_filtered_leave.head());

# emre file
df_emre = pd.read_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/train-3879-3classes.txt", delimiter="~", names=["tweet_id", "user_id", "datetime", "tw_full", "stance"])
print(df_emre.shape);
print(df_emre.head());

df_emre_filtered = df_emre[pd.notnull(df_emre['stance'])]
print(df_emre_filtered.shape);

print(df_emre_filtered.head());
df_emre_filtered.stance = df_emre_filtered.stance.astype(int)

df_emre_filtered_remain = df_emre_filtered[df_emre_filtered['stance']==0]
print(df_emre_filtered_remain.shape);
print(df_emre_filtered_remain.head());

df_emre_filtered_leave = df_emre_filtered[df_emre_filtered['stance']==1]
print(df_emre_filtered_leave.shape);
print(df_emre_filtered_leave.head());

df_emre_filtered_neutral = df_emre_filtered[df_emre_filtered['stance']==2]
print(df_emre_filtered_neutral.shape);
print(df_emre_filtered_neutral.head());

#df_emre_filtered_leaved_sample = df_emre_filtered_leave.sample(735)
#df_emre_filtered_leaved_sample.head();
#df_emre_filtered_leaved_sample.stance=0
#df_emre_filtered_leaved_sample.head();

df_emre_filtered_remain_sample = df_emre_filtered_remain.sample(658)
df_emre_filtered_remain_sample.head();
df_emre_filtered_remain_sample.stance=0
df_emre_filtered_remain_sample.head();

df_emre_filtered_neutral_sample = df_emre_filtered_neutral.sample(658)
df_emre_filtered_neutral_sample.head();
df_emre_filtered_neutral_sample.stance=0
df_emre_filtered_neutral_sample.head();


df_emre_filtered_leave.stance=1

# CREATE REMAIN FILE
df_all_rows_1470_remain_1470_rest = pd.concat([df_emre_filtered_leave, df_emre_filtered_remain_sample, df_emre_filtered_neutral_sample])
df_all_rows_1470_remain_1470_rest.shape;
df_all_rows_1470_remain_1470_rest.head();
df_all_rows_1470_remain_1470_rest_shuffled = df_all_rows_1470_remain_1470_rest.reindex(np.random.permutation(df_all_rows_1470_remain_1470_rest.index))
df_all_rows_1470_remain_1470_rest_shuffled.head();
#df_all_rows_1470_remain_1470_rest_shuffled.to_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/train-leave-1316-rest-1316.txt", columns=None, index=None, sep="~")
df_all_rows_1470_remain_1470_rest_shuffled.to_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/train-leave1316-rest1316.txt", columns=None, index=None, sep="~")


#df_emre_filtered_neutral.stance=1
#print(df_emre_filtered_neutral.head());

del df_emre_filtered_neutral['datetime']
df_emre_filtered_neutral.to_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/train-neutrals.txt", columns=None, index=None, sep="~")


del df_ivan_filtered_remain['unn']
del df_emre_filtered_remain['datetime']

df_all_rows_remain = pd.concat([df_emre_filtered_remain, df_ivan_filtered_remain])
print(df_all_rows_remain.shape);
print(df_all_rows_remain.head());

#df_all_rows_remain.to_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/train-remain-1471-rest-1471-with-730leave-741neutrals.txt", columns=None, index=None, sep="~")


del df_ivan_filtered_leave['unn']
del df_emre_filtered_leave['datetime']

df_all_rows_leave = pd.concat([df_emre_filtered_leave, df_ivan_filtered_leave])
print(df_all_rows_leave.shape);
print(df_all_rows_leave.head());

# df_all_rows_leave.to_csv("/Users/emrecalisir/git/brexit/CSSforPolitics/user_stance/train-leave-1304.txt", columns=None, index=None, sep="~")