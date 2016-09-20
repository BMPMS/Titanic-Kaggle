



def new_features(titan):

    #creates new features

    titan['nofamily3rd'] = 0
    titan.loc[(titan['Pclass']==3) &((titan['Parch']+titan['SibSp'])==0),'nofamily3rd'] = 1
    titan['female1st2nd'] = 0
    titan.loc[(titan['Sex'] == 0) & (titan['Pclass']<3),'female1st2nd'] = 1
    titan['fem_1st_fareover29_under65'] = 0
    titan.loc[(titan['Sex'] == 0) & (titan['Pclass']==1) & (titan['Age'] < 65) & (titan['Fare'] > 29) ,'fem_1st_fareover29_under65'] = 1

    titan['fare_age_combo'] = 0
    titan.loc[(titan['Age'] < 65) & (titan['Fare']>29),'fare_age_combo'] = 1

    return titan
