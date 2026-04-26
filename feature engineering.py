"""
feature_engineering.py
Creates derived features for Titanic dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicFeatureEngineer:
    def __init__(self, df):
        """Initialize with cleaned dataframe"""
        self.df = df.copy()
        self.feature_log = []
        
    def extract_title(self):
        """Extract title from Name column"""
        # Extract title (Mr, Mrs, Miss, etc.)
        self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr',
            'Miss': 'Miss',
            'Mrs': 'Mrs',
            'Master': 'Master',
            'Dr': 'Rare',
            'Rev': 'Rare',
            'Col': 'Rare',
            'Major': 'Rare',
            'Mlle': 'Miss',
            'Countess': 'Rare',
            'Ms': 'Miss',
            'Lady': 'Rare',
            'Jonkheer': 'Rare',
            'Don': 'Rare',
            'Dona': 'Rare',
            'Mme': 'Mrs',
            'Capt': 'Rare',
            'Sir': 'Rare'
        }
        
        self.df['Title'] = self.df['Title'].map(title_mapping)
        self.df['Title'].fillna('Rare', inplace=True)
        
        self.feature_log.append("Created 'Title' feature from Name")
        
        # Visualize title distribution
        print("\nTitle Distribution:")
        print(self.df['Title'].value_counts())
        
        # Survival by title
        title_survival = self.df.groupby('Title')['Survived'].mean()
        print("\nSurvival Rate by Title:")
        print(title_survival)
        
        return self.df
    
    def create_family_features(self):
        """Create family-related features"""
        # Family size (self + siblings/spouse + parents/children)
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        
        # IsAlone flag
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        
        # Family category
        self.df['FamilyCategory'] = pd.cut(
            self.df['FamilySize'],
            bins=[0, 1, 4, 20],
            labels=['Alone', 'Small', 'Large']
        )
        
        self.feature_log.append("Created FamilySize, IsAlone, and FamilyCategory features")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Family size vs survival
        family_survival = self.df.groupby('FamilySize')['Survived'].mean()
        axes[0].bar(family_survival.index, family_survival.values)
        axes[0].set_xlabel('Family Size')
        axes[0].set_ylabel('Survival Rate')
        axes[0].set_title('Survival Rate by Family Size')
        
        # IsAlone vs survival
        alone_survival = self.df.groupby('IsAlone')['Survived'].mean()
        axes[1].bar(['With Family', 'Alone'], alone_survival.values)
        axes[1].set_ylabel('Survival Rate')
        axes[1].set_title('Survival: Alone vs With Family')
        
        plt.tight_layout()
        plt.show()
        
        return self.df
    
    def extract_deck(self):
        """Extract deck from Cabin number"""
        # Extract first letter of cabin as deck
        self.df['Deck'] = self.df['Cabin'].str[0]
        self.df['Deck'] = self.df['Deck'].fillna('Unknown')
        
        # Clean deck values
        valid_decks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
        self.df['Deck'] = self.df['Deck'].apply(lambda x: x if x in valid_decks else 'Unknown')
        
        self.feature_log.append("Extracted 'Deck' feature from Cabin")
        
        # Survival by deck
        deck_survival = self.df.groupby('Deck')['Survived'].mean().sort_values(ascending=False)
        print("\nSurvival Rate by Deck:")
        print(deck_survival)
        
        return self.df
    
    def create_age_groups(self):
        """Create age group categories"""
        # Define age bins
        bins = [0, 12, 18, 35, 60, 100]
        labels = ['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior']
        
        self.df['AgeGroup'] = pd.cut(self.df['Age'], bins=bins, labels=labels)
        
        self.feature_log.append("Created AgeGroup feature")
        
        # Survival by age group
        age_survival = self.df.groupby('AgeGroup')['Survived'].mean()
        print("\nSurvival Rate by Age Group:")
        print(age_survival)
        
        return self.df
    
    def create_fare_per_person(self):
        """Create fare per person feature"""
        self.df['FarePerPerson'] = self.df['Fare'] / self.df['FamilySize']
        
        self.feature_log.append("Created FarePerPerson feature")
        
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features"""
        # Pclass × Fare interaction
        self.df['Pclass_Fare_Interaction'] = self.df['Pclass'] * self.df['FarePerPerson']
        
        # Age × Title interaction
        title_age_map = {
            'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 0
        }
        self.df['Title_Code'] = self.df['Title'].map(title_age_map)
        self.df['Age_Title_Interaction'] = self.df['Age'] * self.df['Title_Code']
        
        self.feature_log.append("Created interaction features: Pclass_Fare_Interaction, Age_Title_Interaction")
        
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        # One-hot encoding for nominal features
        nominal_features = ['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup']
        
        # Handle missing Embarked
        if self.df['Embarked'].isnull().any():
            self.df['Embarked'].fillna('S', inplace=True)
        
        # One-hot encode
        encoded_dfs = []
        for feature in nominal_features:
            if feature in self.df.columns:
                encoded = pd.get_dummies(self.df[feature], prefix=feature, drop_first=False)
                encoded_dfs.append(encoded)
                self.df = pd.concat([self.df, encoded], axis=1)
                self.df.drop(feature, axis=1, inplace=True)
                self.feature_log.append(f"One-hot encoded '{feature}'")
        
        # Ordinal encoding for Pclass
        self.df['Pclass_Ordinal'] = self.df['Pclass']
        
        # Label encoding for FamilyCategory
        family_cat_map = {'Alone': 0, 'Small': 1, 'Large': 2}
        self.df['FamilyCategory_Encoded'] = self.df['FamilyCategory'].map(family_cat_map)
        self.df.drop('FamilyCategory', axis=1, inplace=True)
        
        return self.df
    
    def transform_features(self):
        """Apply log transformations and scaling"""
        # Log transform skewed features
        self.df['Fare_Log'] = np.log1p(self.df['Fare'])
        self.df['FarePerPerson_Log'] = np.log1p(self.df['FarePerPerson'].clip(lower=0))
        
        self.feature_log.append("Applied log transformation to Fare and FarePerPerson")
        
        # Visualize transformations
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        features = [('Fare', 'Fare_Log'), ('FarePerPerson', 'FarePerPerson_Log')]
        
        for i, (orig, transformed) in enumerate(features):
            axes[i, 0].hist(self.df[orig], bins=30, edgecolor='black', alpha=0.7)
            axes[i, 0].set_title(f'{orig} (Original)')
            axes[i, 0].set_xlabel(orig)
            
            axes[i, 1].hist(self.df[transformed], bins=30, edgecolor='black', alpha=0.7, color='green')
            axes[i, 1].set_title(f'{orig} (Log Transformed)')
            axes[i, 1].set_xlabel(transformed)
        
        plt.tight_layout()
        plt.show()
        
        return self.df
    
    def get_feature_summary(self):
        """Print summary of all created features"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING SUMMARY")
        print("=" * 60)
        
        for log in self.feature_log:
            print(f"  ✓ {log}")
        
        print(f"\nTotal features after engineering: {len(self.df.columns)}")
        print(f"Features: {list(self.df.columns)}")
        
        return self.feature_log
    
    def run_full_engineering(self):
        """Execute complete feature engineering pipeline"""
        self.extract_title()
        self.create_family_features()
        self.extract_deck()
        self.create_age_groups()
        self.create_fare_per_person()
        self.create_interaction_features()
        self.encode_categorical()
        self.transform_features()
        self.get_feature_summary()
        
        return self.df

# Execute feature engineering
if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv('train_cleaned.csv')
    
    # Apply feature engineering
    engineer = TitanicFeatureEngineer(df)
    engineered_df = engineer.run_full_engineering()
    
    # Save engineered dataset
    engineered_df.to_csv('train_engineered.csv', index=False)
    print("\nEngineered dataset saved to 'train_engineered.csv'")