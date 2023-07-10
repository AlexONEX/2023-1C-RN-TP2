import pandas as pd

algoritmo = "sanger" #oja_generalizada

input_file = "./"+algoritmo+"/"+algoritmo+"_resultados.txt"
target_file = "./"+algoritmo+"/"+algoritmo+"_resultados_csv.csv"
final_sorted_file = "./"+algoritmo+"/"+algoritmo+"_resultados_csv_sorted.csv"

read_file = pd.read_csv (input_file)
read_file.to_csv (target_file, index=None)

df = pd.read_csv(target_file)
sorted_df = df.sort_values(by=["error"], ascending=True)

sorted_df.to_csv(final_sorted_file, index=False)
