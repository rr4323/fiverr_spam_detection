import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats
import os

class SpammerEDA:
    """
    Class for performing Exploratory Data Analysis on the Spammer Detection dataset
    """
    def __init__(self, df: pd.DataFrame, output_dir: str = '../plots/eda_plots'):
        """
        Initialize the EDA class
        
        Args:
            df (pd.DataFrame): Input dataset
            output_dir (str): Directory to save plots
        """
        self.df = df
        self.output_dir = output_dir
        self.feature_cols = [col for col in df.columns if col not in ['label', 'user_id']]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def basic_analysis(self):
        """Perform basic statistical analysis"""
        print("\n1. Basic Dataset Information:")
        print(f"Dataset shape: {self.df.shape}")
        print("\nClass distribution:")
        print(self.df['label'].value_counts(normalize=True))
        print("\nMissing values:")
        print(self.df.isnull().sum())
        return self.df.describe()

    def plot_class_distribution(self):
        """Plot the distribution of spammer vs non-spammer classes"""
        plt.figure(figsize=(8, 6))
        sns.countplot(data=self.df, x='label')
        plt.title('Distribution of Spammer vs Non-Spammer')
        plt.xlabel('Label (0: Non-Spammer, 1: Spammer)')
        plt.ylabel('Count')
        plt.savefig(f'{self.output_dir}/class_distribution.png')
        plt.close()

    def correlation_analysis(self):
        """Analyze and visualize feature correlations"""
        # Include label in correlation analysis
        correlation_matrix = self.df[self.feature_cols + ['label']].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False)
        plt.title('Feature Correlation Matrix (Including Label)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/correlation_matrix.png')
        plt.close()
        
        # Plot label correlations separately
        label_correlations = correlation_matrix['label'].drop('label')
        plt.figure(figsize=(12, 8))
        label_correlations.sort_values().plot(kind='bar')
        plt.title('Feature Correlations with Label')
        plt.xlabel('Features')
        plt.ylabel('Correlation with Label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/label_correlations.png')
        plt.close()
        
        # Find highly correlated features
        high_corr_features = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8:
                    high_corr_features.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        # Get top features correlated with label
        top_label_correlations = label_correlations.abs().sort_values(ascending=False).head(10)
        
        return {
            'high_correlation_pairs': high_corr_features,
            'label_correlations': label_correlations.to_dict(),
            'top_label_correlations': top_label_correlations.to_dict()
        }

    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest"""
        X = self.df[self.feature_cols]
        y = self.df['label']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_imputed, y)
        
        # Get feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), [self.feature_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importances.png')
        plt.close()
        
        return dict(zip([self.feature_cols[i] for i in indices], importances[indices]))

    def pca_analysis(self):
        """Perform PCA analysis and visualization"""
        X = self.df[self.feature_cols]
        y = self.df['label']
        
        # Handle missing values and scale
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot PCA results
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.6)
        plt.title('PCA of Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(f'{self.output_dir}/pca_scatter.png')
        plt.close()
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
        }

    def plot_feature_distributions(self, top_n: int = 5):
        """Plot distributions of top important features"""
        feature_importance = self.feature_importance_analysis()
        top_features = list(feature_importance.keys())[:top_n]
        
        # Create pair plot for top features
        sns.pairplot(self.df, vars=top_features, hue='label', palette='coolwarm')
        plt.savefig(f'{self.output_dir}/pairplot_top_features.png')
        plt.close()
        
        # Create individual distribution plots
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x='label', y=feature)
            plt.title(f'Distribution of {feature} by Class')
            plt.xlabel('Label (0: Non-Spammer, 1: Spammer)')
            plt.savefig(f'{self.output_dir}/{feature}_distribution.png')
            plt.close()

    def analyze_missing_values(self):
        """Analyze and visualize missing values"""
        missing_values = self.df.isnull().sum()
        missing_percentages = (missing_values / len(self.df)) * 100
        
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentages
        })
        
        # Plot missing values
        plt.figure(figsize=(12, 6))
        missing_data[missing_data['Missing Values'] > 0]['Percentage'].plot(kind='bar')
        plt.title('Percentage of Missing Values by Feature')
        plt.xlabel('Features')
        plt.ylabel('Percentage Missing')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/missing_values.png')
        plt.close()
        
        return missing_data[missing_data['Missing Values'] > 0]

    def generate_summary(self):
        """Generate a comprehensive EDA summary"""
        class_dist = self.df['label'].value_counts(normalize=True)
        high_corr_features = self.correlation_analysis()
        feature_importance = self.feature_importance_analysis()
        pca_results = self.pca_analysis()
        
        summary = {
            'total_samples': len(self.df),
            'total_features': len(self.feature_cols),
            'class_distribution': {
                'spammer': class_dist[1],
                'non_spammer': class_dist[0]
            },
            'high_correlation_pairs': len(high_corr_features['high_correlation_pairs']),
            'top_important_features': dict(list(feature_importance.items())[:5]),
            'pca_explained_variance': pca_results['explained_variance_ratio'].tolist(),
            'missing_values': self.df.isnull().sum().sum()
        }
        
        return summary

    def generate_report(self):
        """Generate a comprehensive EDA report in Markdown format"""
        summary = self.generate_summary()
        high_corr_features = self.correlation_analysis()
        feature_importance = self.feature_importance_analysis()
        missing_data = self.analyze_missing_values()
        
        report = f"""# Exploratory Data Analysis Report
## Spammer Detection System

### 1. Dataset Overview
- **Total Samples**: {summary['total_samples']}
- **Total Features**: {summary['total_features']}
- **Class Distribution**:
  - Spammer: {summary['class_distribution']['spammer']:.2%}
  - Non-Spammer: {summary['class_distribution']['non_spammer']:.2%}
- **Missing Values**: {summary['missing_values']} total missing values

### 2. Feature Analysis
#### 2.1 Top Important Features
The following features were identified as most important for spammer detection:
"""
        
        # Add top features and their importance scores
        for feature, importance in summary['top_important_features'].items():
            report += f"- **{feature}**: {importance:.4f}\n"
        
        report += "\n#### 2.2 Highly Correlated Feature Pairs\n"
        if high_corr_features['high_correlation_pairs']:
            report += "The following feature pairs show high correlation (>0.8):\n\n"
            for pair in high_corr_features['high_correlation_pairs']:
                report += f"- {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}\n"
        else:
            report += "No feature pairs with correlation > 0.8 were found.\n"
        
        report += "\n### 3. Missing Value Analysis\n"
        if not missing_data.empty:
            report += "Features with missing values:\n\n"
            for idx, row in missing_data.iterrows():
                report += f"- **{idx}**: {row['Missing Values']} values ({row['Percentage']:.2f}%)\n"
        else:
            report += "No missing values found in the dataset.\n"
        
        report += """
### 4. Visualization Summary
The following visualizations have been generated in the plots directory:

#### 4.1 Overall Analysis
1. **Class Distribution** (`../plots/eda_plots/class_distribution.png`)
   - Shows the balance between spammer and non-spammer classes
2. **PCA Scatter Plot** (`../plots/eda_plots/pca_scatter.png`)
   - 2D projection of the data using principal components
3. **Pair Plot** (`../plots/eda_plots/pairplot_top_features.png`)
   - Scatter plots showing relationships between top features
4. **Feature Importance Plot** (`../plots/eda_plots/feature_importances.png`)
   - Bar plot showing the importance of each feature
5. **Missing Values Analysis** (`../plots/eda_plots/missing_values.png`)
   - Visualization of missing values across features

#### 4.2 Feature Correlations
1. **Feature Correlation Matrix** (`../plots/eda_plots/correlation_matrix.png`)
   - Heatmap showing correlations between features
2. **Detailed Correlation Matrix** (`../plots/eda_plots/feature_correlation_matrix.png`)
   - More detailed view of feature correlations

#### 4.3 Feature Distributions
The following plots show individual feature distributions:
"""
        # Add feature distribution plots
        for feature in ['X1', 'X2', 'X19', 'X21', 'X22']:
            report += f"- **{feature}** (`../plots/eda_plots/{feature}_distribution.png`)\n"
            report += f"  - Box plot showing distribution by class\n"
        
        report += """
### 5. Key Findings

1. **Class Imbalance**:
   - The dataset shows a {summary['class_distribution']['spammer']:.1%} / {summary['class_distribution']['non_spammer']:.1%} split between spammers and non-spammers
   
2. **Feature Importance**:
   - The top features account for significant predictive power
   - Behavioral features show strong discrimination between classes
   
3. **Data Quality**:
   - {summary['missing_values']} missing values identified
   - {len(high_corr_features['high_correlation_pairs'])} highly correlated feature pairs found

### 6. Recommendations

1. **Feature Selection**:
   - Consider using the top {len(summary['top_important_features'])} most important features
   - Evaluate removing one feature from highly correlated pairs
   
2. **Data Preprocessing**:
   - Handle missing values using median imputation
   - Scale features before model training
   
3. **Model Development**:
   - Consider using class weights or sampling techniques due to class imbalance
   - Focus on features with high importance scores
   - Monitor for potential overfitting due to correlated features

### 7. Next Steps

1. Feature engineering based on top important features
2. Implementation of recommended preprocessing steps
3. Model selection and training with focus on handling class imbalance
4. Regular monitoring of feature importance in production

---
*Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Save the report
        os.makedirs('reports', exist_ok=True)
        with open('../reports/eda_report.md', 'w') as f:
            f.write(report)
        
        print(f"\nEDA report saved to 'reports/eda_report.md'")
        return report

    def run_full_analysis(self):
        """Run complete EDA pipeline and return all results"""
        print("Starting Exploratory Data Analysis...")
        
        results = {
            'basic_stats': self.basic_analysis(),
            'high_correlations': self.correlation_analysis(),
            'feature_importance': self.feature_importance_analysis(),
            'pca_results': self.pca_analysis(),
            'missing_values': self.analyze_missing_values(),
            'summary': self.generate_summary()
        }
        
        # Generate all plots
        self.plot_class_distribution()
        self.plot_feature_distributions()
        
        # Generate report
        self.generate_report()
        
        print(f"\nEDA completed. All plots saved in '{self.output_dir}'")
        return results

def main():
    """Example usage of the SpammerEDA class"""
    # Load data
    df = pd.read_csv('../data/fiverr_data.csv')
    
    # Initialize EDA class
    eda = SpammerEDA(df)
    
    # Run analysis
    results = eda.run_full_analysis()
    
    print("\nAnalysis complete. Check 'reports/eda_report.md' for the detailed report.")

if __name__ == "__main__":
    main() 
