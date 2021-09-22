from TextAnalyzer import TextAnalyzer

text_analyzer = TextAnalyzer("clean_nus_sms.csv")
text_analyzer.clean_data()
text_analyzer.preprocess_text()
text_analyzer.tf_analysis("Singapore")
text_analyzer.tf_analysis("USA")
text_analyzer.tf_idf_analysis("Singapore")
text_analyzer.predict_country_origin()
