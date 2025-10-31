"""
Script para comparar respuestas de diferentes modelos.

Este script carga los archivos de respuestas generados por diferentes modelos
y crea un archivo comparativo para analizar las diferencias.
"""

import pandas as pd
import os
import glob

def find_answer_files(answers_dir):
    """Encuentra todos los archivos de respuestas en el directorio."""
    if not os.path.exists(answers_dir):
        print(f"⚠️  Directorio {answers_dir} no existe")
        return []

    csv_files = glob.glob(os.path.join(answers_dir, "*_answers.csv"))
    return csv_files

def extract_model_name(filename):
    """Extrae el nombre del modelo del nombre del archivo."""
    basename = os.path.basename(filename)
    model_name = basename.replace("_answers.csv", "")
    return model_name

def compare_models(answers_dir, output_file="comparison.csv"):
    """Compara respuestas de diferentes modelos."""

    print("="*60)
    print("Comparador de Respuestas de Modelos")
    print("="*60)
    print()

    # Encontrar archivos
    answer_files = find_answer_files(answers_dir)

    if not answer_files:
        print(f"❌ No se encontraron archivos de respuestas en {answers_dir}")
        return

    print(f"✓ Encontrados {len(answer_files)} archivos de respuestas:")
    for file in answer_files:
        model_name = extract_model_name(file)
        print(f"  - {model_name}")
    print()

    # Cargar primer archivo como base
    base_file = answer_files[0]
    base_model = extract_model_name(base_file)

    print(f"📊 Cargando archivo base: {base_model}")
    comparison_df = pd.read_csv(base_file)

    # Renombrar columna answer
    if 'answer' in comparison_df.columns:
        comparison_df.rename(columns={'answer': f'answer_{base_model}'}, inplace=True)

    # Cargar y fusionar otros archivos
    for answer_file in answer_files[1:]:
        model_name = extract_model_name(answer_file)
        print(f"📊 Agregando: {model_name}")

        try:
            model_df = pd.read_csv(answer_file)

            # Verificar que tenga la misma cantidad de filas
            if len(model_df) != len(comparison_df):
                print(f"⚠️  Advertencia: {model_name} tiene diferente cantidad de filas ({len(model_df)} vs {len(comparison_df)})")
                continue

            # Agregar columna de respuesta
            if 'answer' in model_df.columns:
                comparison_df[f'answer_{model_name}'] = model_df['answer']

        except Exception as e:
            print(f"❌ Error cargando {model_name}: {str(e)}")
            continue

    # Guardar archivo comparativo
    print()
    print(f"💾 Guardando comparación en: {output_file}")
    comparison_df.to_csv(output_file, index=False)

    # Mostrar estadísticas
    print()
    print("="*60)
    print("Estadísticas de la Comparación")
    print("="*60)
    print(f"Total de prompts: {len(comparison_df)}")
    print(f"Total de modelos comparados: {len(answer_files)}")

    # Contar respuestas completas por modelo
    answer_columns = [col for col in comparison_df.columns if col.startswith('answer_')]
    print()
    print("Respuestas completas por modelo:")
    for col in answer_columns:
        model_name = col.replace('answer_', '')
        completed = comparison_df[col].notna().sum()
        percentage = (completed / len(comparison_df)) * 100
        print(f"  {model_name}: {completed}/{len(comparison_df)} ({percentage:.1f}%)")

    # Detectar coincidencias
    if len(answer_columns) >= 2:
        print()
        print("Análisis de coincidencias:")

        # Comparar primeros dos modelos
        col1, col2 = answer_columns[0], answer_columns[1]
        model1 = col1.replace('answer_', '')
        model2 = col2.replace('answer_', '')

        # Respuestas idénticas
        mask_valid = comparison_df[col1].notna() & comparison_df[col2].notna()
        identical = (comparison_df.loc[mask_valid, col1] == comparison_df.loc[mask_valid, col2]).sum()
        total_valid = mask_valid.sum()

        if total_valid > 0:
            percentage = (identical / total_valid) * 100
            print(f"  {model1} vs {model2}:")
            print(f"    Respuestas idénticas: {identical}/{total_valid} ({percentage:.1f}%)")

    print()
    print("="*60)
    print("✅ Comparación completada exitosamente!")
    print(f"📄 Ver archivo: {output_file}")
    print("="*60)

def create_summary_report(comparison_file, output_file="summary_report.txt"):
    """Crea un reporte resumido de la comparación."""

    if not os.path.exists(comparison_file):
        print(f"❌ Archivo {comparison_file} no existe")
        return

    df = pd.read_csv(comparison_file)
    answer_columns = [col for col in df.columns if col.startswith('answer_')]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REPORTE DE COMPARACIÓN DE MODELOS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total de prompts evaluados: {len(df)}\n")
        f.write(f"Total de modelos comparados: {len(answer_columns)}\n\n")

        f.write("-"*80 + "\n")
        f.write("COMPLETITUD POR MODELO\n")
        f.write("-"*80 + "\n\n")

        for col in answer_columns:
            model_name = col.replace('answer_', '')
            completed = df[col].notna().sum()
            errors = df[col].astype(str).str.startswith('ERROR:', na=False).sum()
            successful = completed - errors

            f.write(f"{model_name}:\n")
            f.write(f"  Completadas: {completed}/{len(df)} ({(completed/len(df))*100:.1f}%)\n")
            f.write(f"  Exitosas: {successful}/{len(df)} ({(successful/len(df))*100:.1f}%)\n")
            f.write(f"  Con errores: {errors}/{len(df)} ({(errors/len(df))*100:.1f}%)\n\n")

        # Ejemplos de respuestas
        f.write("-"*80 + "\n")
        f.write("EJEMPLOS DE RESPUESTAS (Primeros 3 prompts)\n")
        f.write("-"*80 + "\n\n")

        for idx in range(min(3, len(df))):
            f.write(f"Prompt {idx + 1}:\n")
            if 'prompt' in df.columns:
                prompt_text = str(df.iloc[idx]['prompt'])
                f.write(f"  {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}\n\n")

            for col in answer_columns:
                model_name = col.replace('answer_', '')
                answer = str(df.iloc[idx][col])
                f.write(f"  [{model_name}]:\n")
                f.write(f"    {answer[:200]}{'...' if len(answer) > 200 else ''}\n\n")

            f.write("-"*80 + "\n\n")

    print(f"✅ Reporte guardado en: {output_file}")

def main():
    """Función principal."""
    import sys

    print()
    print("🔍 Comparador de Respuestas de Modelos")
    print()

    # Determinar directorio de respuestas
    if len(sys.argv) > 1:
        answers_dir = sys.argv[1]
    else:
        # Intentar detectar automáticamente
        if os.path.exists("answers"):
            answers_dir = "answers"
        elif os.path.exists("../answers"):
            answers_dir = "../answers"
        else:
            print("❌ No se pudo encontrar el directorio 'answers'")
            print()
            print("Uso:")
            print(f"  python {os.path.basename(__file__)} [ruta_a_answers]")
            print()
            print("Ejemplo:")
            print(f"  python {os.path.basename(__file__)} answers")
            print(f"  python {os.path.basename(__file__)} ../Ollama/answers")
            return

    # Comparar modelos
    compare_models(answers_dir, "comparison.csv")

    # Crear reporte
    if os.path.exists("comparison.csv"):
        print()
        print("📝 Generando reporte resumido...")
        create_summary_report("comparison.csv", "summary_report.txt")

    print()

if __name__ == "__main__":
    main()

