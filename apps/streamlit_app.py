import os
# Forzar al SDK de Gemini a usar la API v1 estable antes de importar cualquier módulo de google
os.environ["API_VERSION"] = "v1"
os.environ["GOOGLE_API_VERSION"] = "v1"

import sys
from pathlib import Path
import warnings

# 1. Silenciar advertencias de discrepancia de versiones de scikit-learn - Evitar carteles en la UI
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Añadir la carpeta raiz del proyecto al path para poder importar'apps.*'
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import matplotlib
matplotlib.use('Agg')
import matplotlib.axes

# Parche temporal global para evitar ParseException de mathtext en Matplotlib con etiquetas LaTeX de SHAP
original_set_xticklabels = matplotlib.axes.Axes.set_xticklabels
def safe_set_xticklabels(self, labels, *args, **kwargs):
    cleaned_labels = []
    for label in labels:
        if isinstance(label, str):
            cleaned_labels.append(label.replace("$", ""))
        else:
            cleaned_labels.append(label)
    return original_set_xticklabels(self, cleaned_labels, *args, **kwargs)
matplotlib.axes.Axes.set_xticklabels = safe_set_xticklabels

import matplotlib.pyplot as plt
import shap

# Forzar la API v1 estable en las llamadas internas de la SDK
try:
    genai.client.API_VERSION = "v1"
except Exception:
    pass
try:
    genai.client.api_version = "v1"
except Exception:
    pass

# Importar las funciones analiticas reales del repositorio del equipo
from apps.utils import load_model_and_preprocessor, predict_from_inputs

# Configuracion visual de la aplicacion
st.set_page_config(
    page_title="Predicción de Intención de Compra - TIF", 
    page_icon="🛍️",
    layout="wide"
)

# Encabezado institucional (Criterio: Audiencia no tecnica)
st.title("🛍️ Sistema de Inteligencia Artificial: Intención de Compra E-commerce")
st.caption("Tecnicatura Universitaria en Ciencia de Datos e IA Aplicada - UPATECO 2026")
st.markdown("""
Esta interfaz web analiza las métricas de comportamiento y navegación de un cliente en tiempo real.
A través del modelo predictivo **XGBoost Classifier**, el sistema estima la probabilidad de que el usuario concrete su compra.
""")
st.divider()

# Carga de artefactos analiticos con manejo seguro de cache
@st.cache_resource(show_spinner="Cargando el núcleo computacional...")
def load_artifacts():
    model, preprocessor = load_model_and_preprocessor()
    return model, preprocessor

model, preprocessor = load_artifacts()

# Muestra de entrenamiento para SHAP (usada como fondo para probabilidad y para SHAP Global)
@st.cache_resource(show_spinner="Calculando explicabilidad global SHAP (esto tomará solo unos segundos)...")
def get_shap_explainer_and_global_explanation(_model_pipeline, sample_size=100):
    try:
        X_test_path = root / "data" / "processed" / "X_test_processed.csv"
        if X_test_path.exists():
            X_test = pd.read_csv(X_test_path, index_col=0)
            # Tomar una muestra fija/reproducible
            X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
            clf = _model_pipeline.named_steps['clf']
            
            # TreeExplainer usando los datos de test como fondo para obtener probabilidades reales
            explainer = shap.TreeExplainer(clf, data=X_sample, model_output="probability")
            shap_values = explainer(X_sample)
            return explainer, shap_values, X_sample
    except Exception as e:
        st.sidebar.error(f"Error al inicializar SHAP: {e}")
    return None, None, None

explainer, shap_values_global, X_sample_global = get_shap_explainer_and_global_explanation(model)

# Función para construir el medidor visual interactivo (Gauge)
def build_indicator_gauge(prob: float, title: str = "Probabilidad"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 35], 'color': "lightcoral"},
                {'range': [35, 65], 'color': "gold"},
                {'range': [65, 100], 'color': "lightgreen"}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Validacion fisica de los archivos serializados (.pkl)
if model is None:
    st.error("⚠️ Error crítico: El archivo del modelo `modelo_ganador.pkl` no pudo ser hallado en la carpeta `models/`.")
else:
    # Organizacion por pestañas (Negocio vs Criterios Tecnicos)
    tab_simulador, tab_metricas, tab_hibrido = st.tabs(["🔮 Simulador de Sesión", "📊 Evaluación del Modelo", "🤖 Arquitectura Híbrida Conversacional"])

    with tab_simulador:
        st.header("Formulario de Comportamiento del Usuario")
        st.write("Complete los siguientes parámetros lógicos observados en la sesión web:")

        col_inputs1, col_inputs2, col_inputs3 = st.columns(3)

        with col_inputs1:
            st.markdown("### 📈 Páginas Consultadas")
            page_values = st.number_input("Valor de la Página (PageValues)", min_value=0.0, value=0.0, help="Promedio del valor económico de las páginas visitadas antes de finalizar la sesión.")
            product_related = st.number_input("Páginas de Productos", min_value=0, value=12, step=1)
            product_related_duration = st.number_input("Tiempo en Productos (seg)", min_value=0.0, value=350.0)

        with col_inputs2:
            st.markdown("### 🚨 Métricas de Deserción")
            exit_rates = st.slider("Tasa de Salida (Exit Rates)", 0.0, 1.0, 0.02, 0.01)
            bounce_rates = st.slider("Tasa de Rebote (Bounce Rates)", 0.0, 1.0, 0.01, 0.01)
            administrative = st.number_input("Páginas Administrativas", min_value=0, value=2, step=1)

        with col_inputs3:
            st.markdown("### 🗓️ Entorno de la Sesión")
            month = st.selectbox("Mes de Navegación", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=10) # Noviembre por estacionalidad comercial
            visitor_type = st.selectbox("Tipo de Visitante", ["Returning_Visitor", "New_Visitor", "Other"], index=0)
            weekend = st.checkbox("¿Ocurre en Fin de Semana?")

        st.markdown("---")

        if st.button("Analizar Intención de Compra", type="primary"):
            try:
                # Estructuracion completa del diccionario con las 17 llaves nativas del dataset
                inputs_dict = {
                    'Administrative': administrative,
                    'Administrative_Duration': 0.0,
                    'Informational': 0,
                    'Informational_Duration': 0.0,
                    'ProductRelated': product_related,
                    'ProductRelated_Duration': product_related_duration,
                    'BounceRates': bounce_rates,
                    'ExitRates': exit_rates,
                    'PageValues': page_values,
                    'SpecialDay': 0.0,
                    'Month': month,
                    'OperatingSystems': 2,  
                    'Browser': 2,           
                    'Region': 1,            
                    'TrafficType': 1,       
                    'VisitorType': visitor_type,
                    'Weekend': weekend
                }

                # SOLUCIÓN DEL ERROR: Invocación en el orden requerido por utils.py -> (model, preprocessor, inputs_dict)
                prediccion, prob_compra = predict_from_inputs(model, preprocessor, inputs_dict)

                # Guardar en session_state para cruzar datos con la pestaña conversacional (RAG)
                st.session_state['last_prediction'] = {
                    'prediccion': prediccion,
                    'prob_compra': prob_compra,
                    'inputs': inputs_dict
                }

                # Presentacion de resultados con valor explicativo comercial (Exigido en las pautas)
                st.subheader("🎯 Diagnóstico del Algoritmo Predictivo")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if prediccion == 1:
                        st.success("### **Predicción: COMPRA**")
                    else:
                        st.error("### **Predicción: NO COMPRA**")
                    
                    st.metric(label="Confianza / Probabilidad de Conversión", value=f"{prob_compra * 100:.2f}%")

                with col_res2:
                    st.write("**Justificación Conceptual de la Decisión:**")
                    if page_values > 0:
                        st.info(f"💡 **Hallazgo Clave:** El usuario visitó secciones con un `PageValues` de {page_values}. La ingeniería de características demuestra que este es el factor estadístico con mayor peso positivo para concretar transacciones.")
                    else:
                        st.warning("⚠️ **Alerta de Abandono:** El valor económico de la página es cero. Sin interacción con productos de alto interés o carritos activos, la tasa de conversión decrece bruscamente.")
                    
                    if exit_rates > 0.04:
                        st.error(f"📉 **Punto de Fuga:** La tasa de salida observada ({exit_rates * 100}%) supera los límites recomendados, indicando desinterés o fricciones de navegación.")
                    
                    if month == "Nov":
                        st.info("🗓️ **Estacionalidad Promocional:** La sesión se ejecuta dentro de Noviembre, mes históricamente potenciado por campañas anuales de descuento masivo.")

                # Sección Gráfica del Gauge
                st.write("")
                col_gauge1, col_gauge2 = st.columns([1, 1])
                with col_gauge1:
                    st.plotly_chart(build_indicator_gauge(prob_compra, title="Nivel de Intención de Compra"), use_container_width=True)
                with col_gauge2:
                    st.markdown("""
                    **Segmentación de Acciones de Negocio Recomendadas:**
                    * 🔴 **Baja Conversión (0% - 35%):** Navegación casual. No invasiva. Se recomienda retargeting por e-mail a las 24 horas.
                    * 🟡 **Intención Media (35% - 65%):** El usuario evalúa alternativas. Activar un banner interactivo con envío gratis o cupones de descuento por tiempo limitado.
                    * 🟢 **Conversión Inminente (65% - 100%):** Cliente decidido. Optimizar el checkout reduciendo campos de pago para evitar la fricción final.
                    """)
                
                # Explicabilidad local interactiva (SHAP Waterfall Plot)
                if explainer is not None and preprocessor is not None:
                    st.divider()
                    st.subheader("🔍 Explicabilidad en Tiempo Real (¿Por qué el modelo tomó esta decisión?)")
                    st.write("""
                    El gráfico **SHAP Waterfall** de abajo muestra cómo afectó cada comportamiento de navegación a la probabilidad final de compra del cliente.
                    Las barras rojas indican factores que **aumentaron** la intención de compra, y las barras azules indican factores que la **disminuyeron** (medido en variación de probabilidad de 0 a 1).
                    """)
                    
                    try:
                        # 1. Obtener nombres de columnas esperadas
                        if hasattr(preprocessor, 'feature_names_in_'):
                            expected_cols = list(preprocessor.feature_names_in_)
                        else:
                            expected_cols = list(model.feature_names_in_)
                            
                        # 2. Preparar el DataFrame alineado
                        from apps.utils import prepare_input_dataframe
                        df_single = prepare_input_dataframe(inputs_dict, expected_columns=expected_cols, categorical_columns=[])
                        
                        # 3. Transformar con el preprocesador
                        X_single_transformed = preprocessor.transform(df_single)
                        
                        # 4. Obtener nombres legibles de variables transformadas
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feat_names = list(preprocessor.get_feature_names_out())
                            feat_names = [f.split("__")[-1] for f in feat_names]
                        else:
                            feat_names = [f"Var_{i}" for i in range(X_single_transformed.shape[1])]
                            
                        # 5. Crear DataFrame final transformado
                        X_single_df = pd.DataFrame(X_single_transformed, columns=feat_names)
                        
                        # 6. Calcular e individualizar la predicción
                        shap_values_single = explainer(X_single_df)
                        
                        # Graficar en matplotlib de forma no interactiva
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.plots.waterfall(shap_values_single[0], max_display=10, show=False)
                        
                        # Limpiar los símbolos '$' de todos los textos del gráfico para evitar errores del mathtext parser en Windows
                        for t in fig.findobj(lambda x: hasattr(x, 'get_text')):
                            txt = t.get_text()
                            if "$" in txt:
                                t.set_text(txt.replace("$", ""))
                                
                        plt.title("Contribución de Variables a la Probabilidad de Compra (SHAP Local)", fontsize=13, fontweight="bold", pad=15)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e_shap:
                        st.warning(f"No se pudo renderizar el gráfico SHAP local: {e_shap}")

            except Exception as ex:
                st.error(f"Ocurrió un error al procesar la inferencia en el pipeline: {ex}.")

    with tab_metricas:
        st.header("📊 Métricas de Rendimiento y Calidad (XGBoost Classifier)")
        st.write("Estadísticas de validación extraídas del conjunto de prueba independiente:")
        
        # Tarjetas de métricas estilizadas
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(
                '<div style="background-color:rgba(30,144,255,0.08); padding: 18px; border-radius: 8px; border-left: 5px solid #1E90FF; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">'
                '<h5 style="margin:0; font-size:14px; color:#555; font-family:sans-serif;">Exactitud (Accuracy)</h5>'
                '<h2 style="margin:8px 0 0 0; color:#1E90FF; font-size:26px; font-family:sans-serif; font-weight:bold;">88.20%</h2>'
                '</div>', 
                unsafe_allow_html=True
            )
        with col_m2:
            st.markdown(
                '<div style="background-color:rgba(46,139,87,0.08); padding: 18px; border-radius: 8px; border-left: 5px solid #2E8B57; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">'
                '<h5 style="margin:0; font-size:14px; color:#555; font-family:sans-serif;">ROC-AUC</h5>'
                '<h2 style="margin:8px 0 0 0; color:#2E8B57; font-size:26px; font-family:sans-serif; font-weight:bold;">0.9205</h2>'
                '</div>', 
                unsafe_allow_html=True
            )
        with col_m3:
            st.markdown(
                '<div style="background-color:rgba(218,165,32,0.08); padding: 18px; border-radius: 8px; border-left: 5px solid #DAA520; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">'
                '<h5 style="margin:0; font-size:14px; color:#555; font-family:sans-serif;">Sensibilidad (Recall)</h5>'
                '<h2 style="margin:8px 0 0 0; color:#DAA520; font-size:26px; font-family:sans-serif; font-weight:bold;">72.51%</h2>'
                '</div>', 
                unsafe_allow_html=True
            )
        with col_m4:
            st.markdown(
                '<div style="background-color:rgba(255,20,147,0.08); padding: 18px; border-radius: 8px; border-left: 5px solid #FF1493; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">'
                '<h5 style="margin:0; font-size:14px; color:#555; font-family:sans-serif;">F1-Score</h5>'
                '<h2 style="margin:8px 0 0 0; color:#FF1493; font-size:26px; font-family:sans-serif; font-weight:bold;">0.6580</h2>'
                '</div>', 
                unsafe_allow_html=True
            )
            
        # Explicación breve de cada métrica debajo de sus tarjetas
        col_desc1, col_desc2, col_desc3, col_desc4 = st.columns(4)
        with col_desc1:
            st.caption("🎯 **¿Qué es?** Porcentaje total de predicciones correctas (tanto visitas que compran como las que no).")
        with col_desc2:
            st.caption("📈 **¿Qué es?** Capacidad del modelo para distinguir a un comprador de un no comprador (de 0 a 1).")
        with col_desc3:
            st.caption("🔍 **¿Qué es?** Porcentaje de las compras reales que el modelo logró detectar con éxito.")
        with col_desc4:
            st.caption("⚖️ **¿Qué es?** Balance entre precisión (evitar falsas alarmas) y sensibilidad (no perder ventas).")
            
        st.write("")
        st.divider()
        
        # Sub-pestañas para organizar gráficos
        subtab_shap, subtab_importance, subtab_confusion, subtab_curves = st.tabs([
            "🔍 Explicabilidad SHAP Global",
            "📊 Importancia XGBoost",
            "🎯 Matriz de Confusión",
            "📈 Curvas ROC y Comparaciones"
        ])
        
        with subtab_shap:
            st.subheader("🔍 Impacto Global de Características (SHAP Summary Plot)")
            st.write("""
            El siguiente gráfico **SHAP Summary Plot** muestra la contribución de cada variable a nivel global 
            utilizando una muestra representativa del conjunto de test.
            
            * **Eje X (SHAP Value):** Indica si la característica aumentó (derecha) o disminuyó (izquierda) la probabilidad de compra en cada sesión individual.
            * **Color (Feature Value):** Representa el valor relativo de la variable (el **rojo** representa valores altos y el **azul** representa valores bajos).
            * *Ejemplo:* Los puntos rojos agrupados a la derecha en `PageValues` demuestran que valores altos en esta variable empujan fuertemente la predicción hacia **Compra**.
            """)
            if shap_values_global is not None and X_sample_global is not None:
                try:
                    # Limpiar nombres de columnas en la muestra global para el plot
                    X_sample_global_clean = X_sample_global.copy()
                    clean_cols = [c.split("__")[-1] for c in X_sample_global_clean.columns]
                    X_sample_global_clean.columns = clean_cols
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values_global, X_sample_global_clean, show=False)
                    
                    # Limpiar los símbolos '$' de todos los textos del gráfico para evitar errores de mathtext
                    for t in fig.findobj(lambda x: hasattr(x, 'get_text')):
                        txt = t.get_text()
                        if "$" in txt:
                            t.set_text(txt.replace("$", ""))
                            
                    plt.title("Gráfico de Resumen SHAP Global", fontsize=13, fontweight="bold", pad=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e_glob:
                    st.error(f"Ocurrió un error al renderizar el gráfico SHAP global: {e_glob}")
            else:
                st.warning("No se pudieron cargar los datos de test procesados para computar el gráfico SHAP global.")
                
        with subtab_importance:
            st.subheader("📊 Importancia Estructural de Características")
            st.write("""
            Este gráfico representa la importancia de las características basada en la ganancia promedio (Gain) 
            que aporta cada variable al árbol de decisión XGBoost. Muestra cómo divide el modelo la información estructuralmente.
            """)
            im_path = root / 'models' / 'feature_importance_xgboost.png'
            if im_path.exists():
                st.image(str(im_path), caption="Importancia de atributos nativa del modelo XGBoost.", use_container_width=True)
            else:
                st.info("Gráfico de importancia estructural no disponible en la carpeta de modelos.")
                
        with subtab_confusion:
            st.subheader("🎯 Matriz de Confusión")
            st.write("""
            La matriz de confusión evalúa los aciertos y fallas del modelo en el conjunto de test:
            
            * **Verdaderos Positivos (TP - 277):** Compras reales que el modelo identificó correctamente.
            * **Verdaderos Negativos (TN - 1876):** Sesiones sin compra que el modelo clasificó correctamente como NO compra.
            * **Falsos Positivos (FP - 183):** Alertas falsas (el modelo predijo compra pero el cliente no compró).
            * **Falsos Negativos (FN - 105):** Compras reales perdidas (el modelo predijo que no comprarían pero sí compraron).
            """)
            cm_path = root / 'models' / 'confusion_matrix_xgboost.png'
            if cm_path.exists():
                st.image(str(cm_path), caption="Matriz de confusión del modelo ganador (XGBoost).", use_container_width=True)
            else:
                st.info("Gráfico de la matriz de confusión no disponible en la carpeta de modelos.")
                
        with subtab_curves:
            st.subheader("📈 Curvas de Rendimiento (ROC Curves)")
            st.write("""
            La curva ROC y el valor AUC (Área bajo la curva) miden la capacidad de discriminación del modelo a través de distintos umbrales de decisión. Un AUC de 0.92 indica un excelente nivel predictivo.
            """)
            
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                roc_path = root / 'models' / 'roc_curve_xgboost.png'
                if roc_path.exists():
                    st.image(str(roc_path), caption="Curva ROC del XGBoost (Ganador).", use_container_width=True)
                else:
                    st.info("Curva ROC del ganador no disponible.")
            with col_c2:
                comp_path = root / 'models' / 'roc_curves_comparacion.png'
                if comp_path.exists():
                    st.image(str(comp_path), caption="Comparativa de curvas ROC de modelos evaluados.", use_container_width=True)
                else:
                    st.info("Curva comparativa de modelos no disponible.")

    # Intentar obtener la API key de los secretos de Streamlit de forma segura
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
        has_secrets = True
    except Exception:
        gemini_api_key = None
        has_secrets = False

    # Si no está en secretos, habilitar el campo manual en el sidebar
    if not has_secrets:
        if 'gemini_api_key' in st.session_state:
            gemini_api_key = st.session_state['gemini_api_key']
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Configuración del Agente IA")
            st.sidebar.success("✔️ API Key de Gemini configurada de forma temporal.")
        else:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🤖 Configuración del Agente IA")
            user_key = st.sidebar.text_input("Ingresa tu Gemini API Key:", type="password", help="Obtenela gratis en Google AI Studio")
            if user_key:
                gemini_api_key = user_key
                st.session_state['gemini_api_key'] = user_key
                st.rerun()
            st.sidebar.info("💡 Para usar el chat con IA conversacional, ingresa tu API Key.")

    with tab_hibrido:
        st.header("🤖 Ecosistema Híbrido Conversacional RAG")
        
        # Columnas explicativas de la arquitectura (Marco de Integración & Simulación Corporativa)
        col_marco, col_simulacion = st.columns(2)
        with col_marco:
            st.markdown("""
            ### 📋 Marco de Integración
            En sintonía con las directivas de la materia, la evolución de esta aplicación local hacia producción contempla la orquestación de un **Agente RAG híbrido**:
            
            * **Conexión de Datos:** La interfaz recupera de forma directa (In-Memory Retrieval) el contenido de nuestras **políticas comerciales no estructuradas** desde el archivo [politicas_comerciales.txt](file:///c:/Users/Waltter/Desktop/proyectossadsad/Ciencia%20de%20Datos%20e%20Inteligencia%20Artificial%20Aplicada/Modelado/Prediccion%20de%20Intencion%20de%20Compra%20en%20E-commerce/data/politicas_comerciales.txt), inyectándolo dinámicamente como contexto en la ventana del LLM.
            * **Llamada a Funciones (Híbrida):** En lugar de herramientas externas de orquestación complejas, el LLM (*Gemini 2.5 Flash*) interactúa con el estado de la aplicación recuperando las predicciones probabilísticas de nuestro binario **XGBoost** a través de `st.session_state` en tiempo real.
            """)
        with col_simulacion:
            st.markdown("""
            ### 💬 Simulación de Interrogación Corporativa
            Simulá una consulta en lenguaje natural realizada por el gerente de Marketing o del equipo comercial del E-commerce hacia el ecosistema híbrido propuesto:
            
            **Consultas corporativas sugeridas para probar:**
            * *¿Qué opinás del nuevo usuario ingresado?* (ejecutando primero la simulación en la pestaña *Simulador de Sesión*).
            * *¿Qué campaña de marketing o descuento deberíamos aplicar a este cliente en base a su probabilidad de conversión?*
            * *¿Se deben activar pop-ups de salida o reducir campos de pago según nuestras políticas comerciales?*
            """)
        
        st.divider()

        if not gemini_api_key:
            st.warning("⚠️ Se requiere una API Key de Gemini para activar el chat. Por favor, ingresala en el campo correspondiente de la barra lateral.")
        else:
            import os
            os.environ["API_VERSION"] = "v1"
            try:
                genai.configure(api_key=gemini_api_key)
            except Exception as e:
                st.error(f"Error de configuración de la API de Gemini: {e}")
                st.stop()

            # Intentar cargar base de conocimientos
            politicas_path = root / 'data' / 'politicas_comerciales.txt'
            politicas_content = ""
            if politicas_path.exists():
                try:
                    with open(politicas_path, 'r', encoding='utf-8') as f:
                        politicas_content = f.read()
                except Exception as e:
                    st.warning(f"Error al cargar las políticas comerciales: {e}")
            else:
                st.warning("Archivo de políticas comerciales `data/politicas_comerciales.txt` no encontrado.")

            # Mostrar estado de los datos del simulador cruzados
            if 'last_prediction' in st.session_state:
                lp = st.session_state['last_prediction']
                pred_label = "COMPRA" if lp['prediccion'] == 1 else "NO COMPRA"
                st.success(f"📊 **Contexto del Simulador Activo:** Cargados datos de predicción ({pred_label} con {lp['prob_compra'] * 100:.2f}% de probabilidad). El Agente usará esta información en tus preguntas.")
            else:
                st.info("💡 **Consejo:** Hacé una corrida de análisis en la pestaña *Simulador de Sesión* para que el agente converse sobre el perfil de ese cliente en particular.")

            # Inicializar historial del chat
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Contenedor para que el historial de chat quede siempre arriba del input box
            chat_container = st.container()

            # Mostrar historial de chat dentro del contenedor
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Input de chat (se renderiza abajo del contenedor)
            if prompt := st.chat_input("Preguntale al agente (ej. '¿Qué campaña aplico al cliente de la simulación?')"):
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Armar el contexto integrado (RAG + XGBoost)
                context = f"""
                Actúas como un asistente analítico e inteligente especializado en E-commerce y Marketing digital.
                Tu objetivo es asesorar al equipo comercial utilizando la información provista.
                
                --- INICIO BASE DE CONOCIMIENTOS DE LA EMPRESA (RAG) ---
                {politicas_content}
                --- FIN BASE DE CONOCIMIENTOS ---
                
                """

                if 'last_prediction' in st.session_state:
                    lp = st.session_state['last_prediction']
                    pred_label = "COMPRA" if lp['prediccion'] == 1 else "NO COMPRA"
                    context += f"""
                    --- RESULTADOS DE LA ÚLTIMA SIMULACIÓN (XGBOOST) ---
                    - Clasificación del Modelo: El cliente tiene intención de {pred_label}.
                    - Probabilidad de Conversión de Compra: {lp['prob_compra'] * 100:.2f}%
                    - Parámetros detallados de la navegación del cliente:
                      * Valor de la Página (PageValues): {lp['inputs']['PageValues']}
                      * Páginas de Productos visitadas (ProductRelated): {lp['inputs']['ProductRelated']}
                      * Tiempo en Productos en segundos (ProductRelated_Duration): {lp['inputs']['ProductRelated_Duration']}
                      * Tasa de Salida (ExitRates): {lp['inputs']['ExitRates']}
                      * Tasa de Rebote (BounceRates): {lp['inputs']['BounceRates']}
                      * Mes de la Sesión (Month): {lp['inputs']['Month']}
                      * Tipo de Visitante (VisitorType): {lp['inputs']['VisitorType']}
                      * Ocurre en Fin de Semana (Weekend): {lp['inputs']['Weekend']}
                    ----------------------------------------------------
                    """
                else:
                    context += "\n[Nota: Aún no se ha ejecutado ninguna predicción en el simulador para esta sesión.]\n"

                context += """
                INSTRUCCIONES DE RESPUESTA:
                1. Responde con un tono profesional pero cercano, directo y muy fácil de entender. Evita tecnicismos innecesarios (como "XGBoost", "ExitRates" o "BounceRates") y explícalos de forma simple y natural para el equipo de ventas.
                2. Integra las variables del cliente de forma fluida en la redacción, SIN ponerlas entre comillas (por ejemplo, escribe 'productos visitados' o 'probabilidad de abandono' con total naturalidad).
                3. Evita mostrar valores numéricos crudos y abstractos (como decir "un interés de 1.0"). En su lugar, tradúcelos a términos cualitativos (por ejemplo, "interés moderado", "interés alto" o "sin interés comercial"). Las métricas de tiempo, cantidad de páginas y porcentajes sí los puedes citar directamente (ej. "pasó 50 segundos", "visitó 4 páginas", "tasa de salida del 2%").
                4. En lugar de categorías numéricas o técnicas, clasifica la intención de compra del cliente así:
                   - "Cliente muy decidido a comprar" (Intención alta).
                   - "Cliente indeciso o evaluando opciones" (Intención media).
                   - "Cliente que solo está explorando el sitio" (Intención baja).
                5. Usa la Base de Conocimientos para justificar y sugerir acciones comerciales concretas basadas en las políticas de la empresa.
                6. Mantén las respuestas fluidas, amigables y en español.
                
                --- HISTORIAL DE LA CONVERSACIÓN ---
                """

                # Incluir el historial de mensajes anteriores en el contexto
                for msg in st.session_state.messages[:-1]:
                    role_label = "Usuario" if msg["role"] == "user" else "Asistente"
                    context += f"{role_label}: {msg['content']}\n"

                context += f"\nUsuario: {prompt}\nAsistente:"

                try:
                    import requests
                    with chat_container:
                        with st.spinner("Procesando consulta..."):
                            # Llamada directa HTTP a la API v1 de Gemini para evitar bugs de la SDK vieja
                            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={gemini_api_key}"
                            headers = {
                                "Content-Type": "application/json"
                            }
                            payload = {
                                "contents": [
                                    {
                                        "parts": [
                                            {"text": context}
                                        ]
                                    }
                                ]
                            }
                            
                            response = requests.post(url, headers=headers, json=payload)
                            if response.status_code == 200:
                                response_json = response.json()
                                response_text = response_json['candidates'][0]['content']['parts'][0]['text']
                            else:
                                try:
                                    error_msg = response.json()['error']['message']
                                except Exception:
                                    error_msg = response.text
                                raise Exception(f"HTTP {response.status_code}: {error_msg}")

                        with st.chat_message("assistant"):
                            st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error al conectar con Gemini API: {e}")