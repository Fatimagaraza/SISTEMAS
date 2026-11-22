import io
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ==============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="San Francisco Crust Pizza | Cuadro de Mando Ejecutivo",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #F8F9FA;
            border: 1px solid #E6E9EF;
            padding: 14px 16px;
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
        }
        div[data-testid="stMetric"] label {
            color: #4A5568;
            font-weight: 600;
        }
        .ai-card {
            background-color: #FFF9F5;
            border-left: 4px solid #D9381E;
            border-top: 1px solid #F3D9D2;
            border-right: 1px solid #F3D9D2;
            border-bottom: 1px solid #F3D9D2;
            padding: 18px 20px;
            border-radius: 8px;
            margin-top: 10px;
            margin-bottom: 16px;
        }
        .small-muted {
            color: #6C757D;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CATÁLOGO DE SUCURSALES Y GENERACIÓN DE DATOS SINTÉTICOS / MOCKUP
# ==============================================================================
SUCURSALES_CATALOGO: Dict[str, Dict[str, str]] = {
    "SUC-01": {"Sucursal": "Union Square Flagship", "Ciudad": "San Francisco"},
    "SUC-02": {"Sucursal": "Marina District", "Ciudad": "San Francisco"},
    "SUC-03": {"Sucursal": "Mission Bay", "Ciudad": "San Francisco"},
    "SUC-04": {"Sucursal": "Downtown Oakland", "Ciudad": "Oakland"},
    "SUC-05": {"Sucursal": "University Ave", "Ciudad": "Palo Alto"},
    "SUC-06": {"Sucursal": "Santana Row", "Ciudad": "San Jose"},
}


@st.cache_data(show_spinner=False)
def generar_datos_mockup(n_clientes: int = 180, n_ventas: int = 2400, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Genera datos sintéticos coherentes para Clientes, Productos y Ventas."""
    rng = np.random.default_rng(seed)

    # 1. PRODUCTOS (Productos: ProductoID, Categoria, Producto, Tamaño, CostoUSD, PrecioVentaUSD, MargenBrutoUSD, MargenPorcentaje, Activo)
    productos_base = [
        ("PRD-01", "Pizzas", "Sourdough Margherita", "Mediana (12\")", 3.40, 14.50, True),
        ("PRD-02", "Pizzas", "Sourdough Margherita", "Familiar (16\")", 4.80, 19.50, True),
        ("PRD-03", "Pizzas", "Bay Area Pepperoni", "Mediana (12\")", 4.10, 16.00, True),
        ("PRD-04", "Pizzas", "Bay Area Pepperoni", "Familiar (16\")", 5.75, 21.50, True),
        ("PRD-05", "Pizzas", "Wild Mushroom & Truffle", "Mediana (12\")", 5.20, 18.00, True),
        ("PRD-06", "Pizzas", "Spicy Honey Soppressata", "Familiar (16\")", 6.30, 22.90, True),
        ("PRD-07", "Entradas", "Garlic Knot Basket", "Estándar", 1.15, 7.50, True),
        ("PRD-08", "Entradas", "Burrata & Heirloom Salad", "Estándar", 3.60, 12.50, True),
        ("PRD-09", "Entradas", "Crispy Calamari Fritti", "Estándar", 4.40, 13.90, True),
        ("PRD-10", "Bebidas", "Craft Cola / Sodas", "Mediana", 0.55, 3.90, True),
        ("PRD-11", "Bebidas", "Local IPA Draft", "Pinta", 1.60, 7.90, True),
        ("PRD-12", "Bebidas", "San Pellegrino Sparkling", "Botella 500ml", 0.90, 4.20, True),
        ("PRD-13", "Postres", "Classic Tiramisu Cup", "Individual", 1.85, 7.90, True),
        ("PRD-14", "Postres", "Cannoli Siciliano Duo", "2 Unidades", 1.70, 7.20, True),
    ]

    df_productos = pd.DataFrame(
        productos_base,
        columns=["ProductoID", "Categoria", "Producto", "Tamaño", "CostoUSD", "PrecioVentaUSD", "Activo"],
    )
    df_productos["CostoUSD"] = df_productos["CostoUSD"].astype(float)
    df_productos["PrecioVentaUSD"] = df_productos["PrecioVentaUSD"].astype(float)
    df_productos["MargenBrutoUSD"] = (df_productos["PrecioVentaUSD"] - df_productos["CostoUSD"]).round(2)
    df_productos["MargenPorcentaje"] = ((df_productos["MargenBrutoUSD"] / df_productos["PrecioVentaUSD"]) * 100).round(2)

    # 2. CLIENTES (Clientes: ClienteID, Nombre, Sexo, Edad, Ciudad, Segmento, CanalFavorito, ClienteActivo, NPS)
    nombres = [
        "Sofía Martínez", "Carlos Mendoza", "Elena Rodríguez", "Mateo Silva", "Valeria Castro",
        "Diego Morales", "Camila Herrera", "Lucas Navarro", "Isabella Vega", "Sebastián Ortiz",
        "Lucía Ramos", "Alejandro Reyes", "Martina Cruz", "Daniel Flores", "Antonella Díaz",
        "Gabriel Romero", "Paula Morales", "Nicolás Ruiz", "Fernanda Jiménez", "Julián Gómez",
    ]
    apellidos = [
        "Chen", "Patel", "Kim", "García", "Alvarado", "Bautista", "Serrano", "Molina",
        "Delgado", "Castillo", "Rojas", "Guerrero", "Mendoza", "Paredes", "Vargas",
    ]
    ciudades_list = ["San Francisco", "Oakland", "Palo Alto", "San Jose"]
    segmentos_list = ["Corporativo", "Familiar", "Profesional", "Estudiante"]
    canales_list = ["App Móvil", "Web", "Salón", "Delivery Terceros"]
    sexos_list = ["F", "M", "NB"]

    clientes_rows = []
    for i in range(1, n_clientes + 1):
        cid = f"CLI-{i:04d}"
        nombre_completo = f"{rng.choice(nombres)} {rng.choice(apellidos)}"
        sexo = str(rng.choice(sexos_list, p=[0.47, 0.48, 0.05]))
        segmento = str(rng.choice(segmentos_list, p=[0.24, 0.28, 0.28, 0.20]))

        if segmento == "Estudiante":
            edad = int(rng.integers(18, 26))
            ciudad = str(rng.choice(["San Francisco", "Oakland", "Palo Alto"], p=[0.45, 0.30, 0.25]))
            canal = str(rng.choice(canales_list, p=[0.45, 0.15, 0.20, 0.20]))
        elif segmento == "Corporativo":
            edad = int(rng.integers(27, 56))
            ciudad = str(rng.choice(["San Francisco", "Palo Alto", "San Jose", "Oakland"], p=[0.45, 0.25, 0.20, 0.10]))
            canal = str(rng.choice(canales_list, p=[0.20, 0.40, 0.20, 0.20]))
        elif segmento == "Familiar":
            edad = int(rng.integers(30, 62))
            ciudad = str(rng.choice(["San Jose", "Oakland", "San Francisco", "Palo Alto"], p=[0.35, 0.25, 0.25, 0.15]))
            canal = str(rng.choice(canales_list, p=[0.25, 0.25, 0.35, 0.15]))
        else:  # Profesional
            edad = int(rng.integers(24, 50))
            ciudad = str(rng.choice(ciudades_list, p=[0.40, 0.20, 0.20, 0.20]))
            canal = str(rng.choice(canales_list, p=[0.38, 0.27, 0.20, 0.15]))

        activo = bool(rng.choice([True, False], p=[0.86, 0.14]))
        nps = int(np.clip(rng.normal(loc=7.8 if activo else 5.6, scale=1.8), 0, 10))

        clientes_rows.append(
            {
                "ClienteID": cid,
                "Nombre": nombre_completo,
                "Sexo": sexo,
                "Edad": edad,
                "Ciudad": ciudad,
                "Segmento": segmento,
                "CanalFavorito": canal,
                "ClienteActivo": activo,
                "NPS": nps,
            }
        )

    df_clientes = pd.DataFrame(clientes_rows)

    # 3. VENTAS (Ventas: ID_Transaccion, Fecha, ID_Cliente, ID_Producto, Cantidad, ID_Sucursal, Metodo_Pago)
    fecha_fin = pd.Timestamp.today().normalize()
    fecha_inicio = fecha_fin - pd.Timedelta(days=119)  # ~4 meses de historia reciente
    rango_fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")

    # Ponderación estacional suave (más volumen viernes y fines de semana)
    pesos_dias = np.array([1.35 if d.dayofweek in (4, 5, 6) else 0.90 for d in rango_fechas])
    pesos_dias = pesos_dias / pesos_dias.sum()

    fechas_elegidas = rng.choice(rango_fechas, size=n_ventas, p=pesos_dias)
    ids_clientes = rng.choice(df_clientes["ClienteID"].values, size=n_ventas)
    mapa_cliente_ciudad = df_clientes.set_index("ClienteID")["Ciudad"].to_dict()
    mapa_cliente_seg = df_clientes.set_index("ClienteID")["Segmento"].to_dict()

    sucursales_ids = list(SUCURSALES_CATALOGO.keys())
    productos_ids = df_productos["ProductoID"].values
    pesos_productos = np.array([0.14, 0.11, 0.15, 0.12, 0.08, 0.07, 0.07, 0.04, 0.03, 0.07, 0.04, 0.03, 0.02, 0.02])
    pesos_productos = pesos_productos / pesos_productos.sum()

    metodos_pago = ["Tarjeta Crédito/Débito", "Apple Pay / Google Pay", "Efectivo", "Transferencia / App"]
    pesos_pago = [0.46, 0.34, 0.11, 0.09]

    ventas_rows = []
    for idx in range(n_ventas):
        cid = str(ids_clientes[idx])
        ciudad_cli = mapa_cliente_ciudad.get(cid, "San Francisco")
        seg_cli = mapa_cliente_seg.get(cid, "Profesional")

        # Elegir sucursal alineada preferentemente con la ciudad del cliente
        candidatas_suc = [k for k, v in SUCURSALES_CATALOGO.items() if v["Ciudad"] == ciudad_cli]
        if candidatas_suc and rng.random() < 0.78:
            sid = str(rng.choice(candidatas_suc))
        else:
            sid = str(rng.choice(sucursales_ids))

        pid = str(rng.choice(productos_ids, p=pesos_productos))

        if seg_cli == "Corporativo":
            cantidad = int(rng.choice([1, 2, 3, 4, 5], p=[0.25, 0.30, 0.22, 0.13, 0.10]))
        elif seg_cli == "Familiar":
            cantidad = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.40, 0.23, 0.12]))
        else:
            cantidad = int(rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10]))

        ventas_rows.append(
            {
                "ID_Transaccion": f"TRX-{idx + 1:05d}",
                "Fecha": pd.Timestamp(fechas_elegidas[idx]),
                "ID_Cliente": cid,
                "ID_Producto": pid,
                "Cantidad": cantidad,
                "ID_Sucursal": sid,
                "Metodo_Pago": str(rng.choice(metodos_pago, p=pesos_pago)),
            }
        )

    df_ventas = pd.DataFrame(ventas_rows).sort_values("Fecha").reset_index(drop=True)
    return df_ventas, df_clientes, df_productos


# ==============================================================================
# CARGA DINÁMICA, NORMALIZACIÓN Y ENRIQUECIMIENTO DE DATOS
# ==============================================================================
def _leer_archivo_subido(uploaded_file) -> pd.DataFrame:
    """Lee un archivo CSV o Excel desde st.file_uploader."""
    nombre = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    if nombre.endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")
    elif nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes))
    else:
        raise ValueError("Formato no soportado. Sube un archivo .csv, .xlsx o .xls.")


def procesar_ventas_subidas(
    uploaded_file,
    df_clientes_base: pd.DataFrame,
    df_productos_base: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Valida y normaliza un archivo de ventas externo.
    Soporta tanto el esquema exacto de Ventas como columnas opcionales
    (Ciudad, Sucursal, Segmento, Categoria, Producto, PrecioVentaUSD, CostoUSD)
    para evitar errores en demostraciones en vivo.
    """
    avisos: List[str] = []
    df_raw = _leer_archivo_subido(uploaded_file)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # Alias comunes por si la hoja de cálculo usa nombres ligeramente distintos
    alias_map = {
        "transaccion_id": "ID_Transaccion",
        "id": "ID_Transaccion",
        "date": "Fecha",
        "fecha_venta": "Fecha",
        "cliente_id": "ID_Cliente",
        "idcliente": "ID_Cliente",
        "producto_id": "ID_Producto",
        "idproducto": "ID_Producto",
        "qty": "Cantidad",
        "unidades": "Cantidad",
        "sucursal_id": "ID_Sucursal",
        "idsucursal": "ID_Sucursal",
        "metodo_pago": "Metodo_Pago",
        "pago": "Metodo_Pago",
    }
    renombres = {c: alias_map[c.lower()] for c in df_raw.columns if c.lower() in alias_map and alias_map[c.lower()] not in df_raw.columns}
    if renombres:
        df_raw = df_raw.rename(columns=renombres)

    requeridas = ["Fecha", "ID_Producto", "Cantidad"]
    faltantes = [col for col in requeridas if col not in df_raw.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias en el archivo: {', '.join(faltantes)}")

    df_v = df_raw.copy()

    if "ID_Transaccion" not in df_v.columns:
        df_v["ID_Transaccion"] = [f"UP-{i + 1:05d}" for i in range(len(df_v))]
        avisos.append("No se encontró 'ID_Transaccion'; se generaron identificadores automáticos.")

    df_v["Fecha"] = pd.to_datetime(df_v["Fecha"], errors="coerce")
    n_fechas_invalidas = int(df_v["Fecha"].isna().sum())
    if n_fechas_invalidas > 0:
        df_v = df_v.dropna(subset=["Fecha"])
        avisos.append(f"Se descartaron {n_fechas_invalidas} fila(s) con fecha inválida.")

    df_v["Cantidad"] = pd.to_numeric(df_v["Cantidad"], errors="coerce").fillna(1).clip(lower=1).astype(int)

    if "ID_Cliente" not in df_v.columns:
        clientes_pool = df_clientes_base["ClienteID"].tolist()
        df_v["ID_Cliente"] = [clientes_pool[i % len(clientes_pool)] for i in range(len(df_v))]
        avisos.append("No se encontró 'ID_Cliente'; se asignaron clientes del catálogo base.")
    else:
        df_v["ID_Cliente"] = df_v["ID_Cliente"].astype(str).str.strip()

    if "ID_Sucursal" not in df_v.columns:
        suc_pool = list(SUCURSALES_CATALOGO.keys())
        df_v["ID_Sucursal"] = [suc_pool[i % len(suc_pool)] for i in range(len(df_v))]
        avisos.append("No se encontró 'ID_Sucursal'; se asignaron sucursales por defecto.")
    else:
        df_v["ID_Sucursal"] = df_v["ID_Sucursal"].astype(str).str.strip()

    if "Metodo_Pago" not in df_v.columns:
        df_v["Metodo_Pago"] = "Tarjeta Crédito/Débito"

    df_v["ID_Producto"] = df_v["ID_Producto"].astype(str).str.strip()

    # Completar clientes nuevos si aparecen ID_Cliente no registrados en df_clientes_base
    ids_conocidos = set(df_clientes_base["ClienteID"].astype(str))
    nuevos_clientes = sorted(set(df_v["ID_Cliente"]) - ids_conocidos)
    df_clientes_out = df_clientes_base.copy()
    if nuevos_clientes:
        ciudades_def = ["San Francisco", "Oakland", "Palo Alto", "San Jose"]
        segs_def = ["Corporativo", "Familiar", "Profesional", "Estudiante"]
        extra_cli = []
        for idx, cid in enumerate(nuevos_clientes):
            extra_cli.append(
                {
                    "ClienteID": cid,
                    "Nombre": f"Cliente {cid}",
                    "Sexo": "ND",
                    "Edad": 34,
                    "Ciudad": ciudades_def[idx % len(ciudades_def)],
                    "Segmento": segs_def[idx % len(segs_def)],
                    "CanalFavorito": "Web",
                    "ClienteActivo": True,
                    "NPS": 8,
                }
            )
        df_clientes_out = pd.concat([df_clientes_out, pd.DataFrame(extra_cli)], ignore_index=True)
        avisos.append(f"Se registraron {len(nuevos_clientes)} cliente(s) nuevo(s) detectados en el archivo.")

    # Si el archivo incluye columnas de producto propias (PrecioVentaUSD / CostoUSD), ajustar catálogo
    df_prod_out = df_productos_base.copy()
    ids_prod_conocidos = set(df_prod_out["ProductoID"].astype(str))
    nuevos_prod = sorted(set(df_v["ID_Producto"]) - ids_prod_conocidos)
    if nuevos_prod:
        extra_prod = []
        for idx, pid in enumerate(nuevos_prod):
            nombre_p = str(df_v.loc[df_v["ID_Producto"] == pid, "Producto"].iloc[0]) if "Producto" in df_v.columns else f"Ítem {pid}"
            cat_p = str(df_v.loc[df_v["ID_Producto"] == pid, "Categoria"].iloc[0]) if "Categoria" in df_v.columns else "Pizzas"
            tam_p = str(df_v.loc[df_v["ID_Producto"] == pid, "Tamaño"].iloc[0]) if "Tamaño" in df_v.columns else "Mediana (12\")"
            precio_p = float(pd.to_numeric(df_v.loc[df_v["ID_Producto"] == pid, "PrecioVentaUSD"], errors="coerce").dropna().mean()) if "PrecioVentaUSD" in df_v.columns else 15.0
            costo_p = float(pd.to_numeric(df_v.loc[df_v["ID_Producto"] == pid, "CostoUSD"], errors="coerce").dropna().mean()) if "CostoUSD" in df_v.columns else round(precio_p * 0.32, 2)
            margen_usd = round(precio_p - costo_p, 2)
            margen_pct = round((margen_usd / precio_p) * 100, 2) if precio_p > 0 else 0.0
            extra_prod.append(
                {
                    "ProductoID": pid,
                    "Categoria": cat_p,
                    "Producto": nombre_p,
                    "Tamaño": tam_p,
                    "CostoUSD": costo_p,
                    "PrecioVentaUSD": precio_p,
                    "MargenBrutoUSD": margen_usd,
                    "MargenPorcentaje": margen_pct,
                    "Activo": True,
                }
            )
        df_prod_out = pd.concat([df_prod_out, pd.DataFrame(extra_prod)], ignore_index=True)
        avisos.append(f"Se añadieron {len(nuevos_prod)} producto(s) nuevo(s) al catálogo.")

    return df_v, df_clientes_out, df_prod_out, avisos


def enriquecer_datos(df_ventas: pd.DataFrame, df_clientes: pd.DataFrame, df_productos: pd.DataFrame) -> pd.DataFrame:
    """Une Ventas + Clientes + Productos + Sucursales y calcula métricas por línea."""
    df_suc = pd.DataFrame(
        [{"ID_Sucursal": k, "Sucursal": v["Sucursal"], "Ciudad_Sucursal": v["Ciudad"]} for k, v in SUCURSALES_CATALOGO.items()]
    )

    df = df_ventas.merge(df_clientes, left_on="ID_Cliente", right_on="ClienteID", how="left")
    df = df.merge(df_productos, left_on="ID_Producto", right_on="ProductoID", how="left")
    df = df.merge(df_suc, on="ID_Sucursal", how="left")

    # Completar valores faltantes de forma segura
    df["Ciudad"] = df["Ciudad"].fillna(df["Ciudad_Sucursal"]).fillna("San Francisco")
    df["Sucursal"] = df["Sucursal"].fillna(df["ID_Sucursal"])
    df["Segmento"] = df["Segmento"].fillna("Profesional")
    df["Categoria"] = df["Categoria"].fillna("Pizzas")
    df["Producto"] = df["Producto"].fillna(df["ID_Producto"])
    df["Tamaño"] = df["Tamaño"].fillna("Estándar")
    df["PrecioVentaUSD"] = pd.to_numeric(df["PrecioVentaUSD"], errors="coerce").fillna(15.0)
    df["CostoUSD"] = pd.to_numeric(df["CostoUSD"], errors="coerce").fillna(4.8)
    df["Cantidad"] = pd.to_numeric(df["Cantidad"], errors="coerce").fillna(1).astype(int)

    df["Venta_Total_USD"] = (df["Cantidad"] * df["PrecioVentaUSD"]).round(2)
    df["Costo_Total_USD"] = (df["Cantidad"] * df["CostoUSD"]).round(2)
    df["Utilidad_Bruta_USD"] = (df["Venta_Total_USD"] - df["Costo_Total_USD"]).round(2)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Etiqueta_Producto"] = df["Producto"] + " (" + df["Tamaño"] + ")"

    return df


# ==============================================================================
# ASISTENTE DE IA COMERCIAL (OPENAI + FALLBACK DETERMINISTA)
# ==============================================================================
def construir_resumen_ejecutivo(df_filt: pd.DataFrame) -> Dict[str, object]:
    """Extrae KPIs y desgloses clave del conjunto de datos actualmente filtrado."""
    ventas_totales = float(df_filt["Venta_Total_USD"].sum())
    utilidad_bruta = float(df_filt["Utilidad_Bruta_USD"].sum())
    margen_pct = (utilidad_bruta / ventas_totales * 100) if ventas_totales > 0 else 0.0
    n_transacciones = int(df_filt["ID_Transaccion"].nunique()) if "ID_Transaccion" in df_filt.columns else len(df_filt)
    ticket_promedio = (ventas_totales / n_transacciones) if n_transacciones > 0 else 0.0
    clientes_unicos = int(df_filt["ID_Cliente"].nunique()) if "ID_Cliente" in df_filt.columns else 0

    por_prod = (
        df_filt.groupby(["Producto", "Categoria"], as_index=False)
        .agg(Ventas=("Venta_Total_USD", "sum"), Utilidad=("Utilidad_Bruta_USD", "sum"), Unidades=("Cantidad", "sum"))
        .sort_values("Ventas", ascending=False)
    )
    por_cat = (
        df_filt.groupby("Categoria", as_index=False)
        .agg(Ventas=("Venta_Total_USD", "sum"), Utilidad=("Utilidad_Bruta_USD", "sum"))
    )
    por_cat["Margen_Pct"] = np.where(por_cat["Ventas"] > 0, por_cat["Utilidad"] / por_cat["Ventas"] * 100, 0.0)

    por_suc = (
        df_filt.groupby(["Sucursal", "Ciudad"], as_index=False)
        .agg(Ventas=("Venta_Total_USD", "sum"), Utilidad=("Utilidad_Bruta_USD", "sum"))
        .sort_values("Ventas", ascending=False)
    )

    por_seg = (
        df_filt.groupby("Segmento", as_index=False)
        .agg(Ventas=("Venta_Total_USD", "sum"), Utilidad=("Utilidad_Bruta_USD", "sum"), Transacciones=("ID_Transaccion", "nunique"))
        .sort_values("Ventas", ascending=False)
    )

    return {
        "ventas_totales": ventas_totales,
        "utilidad_bruta": utilidad_bruta,
        "margen_pct": margen_pct,
        "ticket_promedio": ticket_promedio,
        "clientes_unicos": clientes_unicos,
        "n_transacciones": n_transacciones,
        "top_productos": por_prod.head(5).to_dict(orient="records"),
        "categorias": por_cat.to_dict(orient="records"),
        "sucursales": por_suc.to_dict(orient="records"),
        "segmentos": por_seg.to_dict(orient="records"),
    }


def generar_analisis_fallback(resumen: Dict[str, object]) -> str:
    """Motor de reglas comerciales determinista: funciona siempre sin necesidad de API key."""
    ventas = resumen["ventas_totales"]
    utilidad = resumen["utilidad_bruta"]
    margen = resumen["margen_pct"]
    ticket = resumen["ticket_promedio"]
    clientes = resumen["clientes_unicos"]
    trx = resumen["n_transacciones"]

    top_prods = resumen["top_productos"]
    cats = resumen["categorias"]
    sucs = resumen["sucursales"]
    segs = resumen["segmentos"]

    lider_prod = top_prods[0]["Producto"] if top_prods else "N/A"
    lider_prod_ventas = top_prods[0]["Ventas"] if top_prods else 0.0

    lider_suc = sucs[0]["Sucursal"] if sucs else "N/A"
    rezagada_suc = sucs[-1]["Sucursal"] if len(sucs) > 1 else lider_suc
    rezagada_suc_ventas = sucs[-1]["Ventas"] if len(sucs) > 1 else 0.0

    lider_seg = segs[0]["Segmento"] if segs else "N/A"
    lider_seg_pct = (segs[0]["Ventas"] / ventas * 100) if (segs and ventas > 0) else 0.0

    # Categoría con mayor y menor margen
    cats_sorted_margen = sorted(cats, key=lambda x: x["Margen_Pct"], reverse=True)
    mejor_cat = cats_sorted_margen[0]["Categoria"] if cats_sorted_margen else "Bebidas"
    menor_cat = cats_sorted_margen[-1]["Categoria"] if cats_sorted_margen else "Entradas"

    # Regla sobre ticket promedio y margen bruto
    if margen >= 68.0:
        diagnostico_margen = f"El margen bruto es saludable (**{margen:.1f}%**), lo que da holgura para promociones de volumen sin erosionar la rentabilidad."
    elif margen >= 62.0:
        diagnostico_margen = f"El margen bruto se ubica en un rango competitivo (**{margen:.1f}%**), aunque sensible a variaciones en el costo de insumos."
    else:
        diagnostico_margen = f"El margen bruto está ajustado (**{margen:.1f}%**); conviene revisar recetas, mermas y precios en categorías de menor rendimiento."

    recs: List[str] = [
        f"**Impulsar venta cruzada hacia {mejor_cat}:** {diagnostico_margen} Configura combos de pizza + {mejor_cat.lower()} en caja y app para elevar el ticket promedio actual de **${ticket:,.2f}**.",
        f"**Capitalizar el segmento {lider_seg} ({lider_seg_pct:.1f}% de las ventas):** Diseña ofertas específicas para **{lider_seg}** (por ejemplo, packs de almuerzo o familiares recurrentes) y mide recompra en las próximas 4 semanas.",
        f"**Plan de choque en {rezagada_suc} vs. benchmark de {lider_suc}:** **{lider_suc}** lidera el periodo, mientras **{rezagada_suc}** registra **${rezagada_suc_ventas:,.2f}**. Audita horarios pico, disponibilidad en canales digitales y promociones locales en **{rezagada_suc}**.",
        f"**Proteger disponibilidad y visibilidad de {lider_prod}:** Es el producto estrella (**${lider_prod_ventas:,.2f}**). Asegura stock de seguridad en ingredientes clave y destaca variantes de mayor margen en el menú.",
    ]

    if ticket < 28.0:
        recs.append(
            f"**Elevar el ticket promedio (actualmente ${ticket:,.2f}):** Introduce un umbral de envío gratis o beneficio de entrada/postre a partir de **${max(28.0, round(ticket * 1.25, 0)):,.0f}**."
        )

    bullets_md = "\n".join([f"- {r}" for r in recs])

    return (
        f"### Diagnóstico Ejecutivo (Modo Reglas Comerciales)\n"
        f"En el periodo filtrado se registraron **${ventas:,.2f}** en ventas brutas y **${utilidad:,.2f}** "
        f"en utilidad bruta (**{margen:.1f}%** de margen) a través de **{trx:,}** transacciones y **{clientes:,}** clientes únicos.\n\n"
        f"### Recomendaciones Accionables\n"
        f"{bullets_md}\n\n"
        f"<span class='small-muted'>*Nota: Generado automáticamente mediante reglas comerciales de negocio (sin dependencia de API externa).*</span>"
    )


def generar_analisis_ia(resumen: Dict[str, object], api_key: str, modelo: str = "gpt-4o-mini") -> Tuple[str, str]:
    """
    Intenta generar un análisis con OpenAI si se proporciona API Key válida;
    en caso contrario (o ante cualquier error de red/cuota), activa el fallback automático.
    Retorna (texto_markdown, modo_usado).
    """
    clave_limpia = (api_key or "").strip()
    if not clave_limpia or OpenAI is None:
        return generar_analisis_fallback(resumen), "Reglas Comerciales (Fallback)"

    try:
        cliente = OpenAI(api_key=clave_limpia)
        prompt_usuario = (
            "Eres un consultor senior de BI y estrategia comercial para la cadena de pizzerías "
            "'San Francisco Crust Pizza'. Analiza las siguientes métricas filtradas y entrega:\n"
            "1) Un breve diagnóstico ejecutivo (máx. 3 líneas).\n"
            "2) 4 recomendaciones comerciales concretas, cuantificadas y accionables en español.\n\n"
            f"Datos resumidos: {resumen}"
        )
        respuesta = cliente.chat.completions.create(
            model=modelo,
            temperature=0.35,
            max_tokens=650,
            messages=[
                {"role": "system", "content": "Responde siempre en español claro, ejecutivo y orientado a rentabilidad."},
                {"role": "user", "content": prompt_usuario},
            ],
        )
        contenido = (respuesta.choices[0].message.content or "").strip()
        if not contenido:
            return generar_analisis_fallback(resumen), "Reglas Comerciales (Fallback)"
        return contenido, f"OpenAI ({modelo})"
    except Exception as exc:
        texto_fb = generar_analisis_fallback(resumen)
        aviso_err = f"\n\n> *Aviso en vivo: No se pudo conectar con OpenAI ({type(exc).__name__}). Se aplicó automáticamente el modo de reglas comerciales para garantizar continuidad.*"
        return texto_fb + aviso_err, "Reglas Comerciales (Fallback por error de API)"


# ==============================================================================
# INICIALIZACIÓN DE ESTADO EN SESSION_STATE
# ==============================================================================
if "df_ventas_base" not in st.session_state:
    v_init, c_init, p_init = generar_datos_mockup()
    st.session_state["df_ventas_base"] = v_init
    st.session_state["df_clientes_base"] = c_init
    st.session_state["df_productos_base"] = p_init
    st.session_state["origen_datos"] = "Datos Sintéticos por Defecto (San Francisco Crust Pizza)"
    st.session_state["avisos_carga"] = []

if "ia_resultado" not in st.session_state:
    st.session_state["ia_resultado"] = None
    st.session_state["ia_modo"] = None


# ==============================================================================
# BARRA LATERAL: CARGA DINÁMICA, FILTROS Y CONFIGURACIÓN DE IA
# ==============================================================================
with st.sidebar:
    st.title("🍕 SF Crust Pizza")
    st.caption("Panel de Control Comercial y BI")

    # 1. CARGA DINÁMICA DE DATOS
    st.subheader("1. Carga de Datos de Ventas")
    archivo_subido = st.file_uploader(
        "Sube archivo de ventas (CSV o Excel)",
        type=["csv", "xlsx", "xls"],
        help="Columnas esperadas: ID_Transaccion, Fecha, ID_Cliente, ID_Producto, Cantidad, ID_Sucursal, Metodo_Pago",
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Restaurar Mockup", use_container_width=True):
            v_def, c_def, p_def = generar_datos_mockup()
            st.session_state["df_ventas_base"] = v_def
            st.session_state["df_clientes_base"] = c_def
            st.session_state["df_productos_base"] = p_def
            st.session_state["origen_datos"] = "Datos Sintéticos por Defecto (San Francisco Crust Pizza)"
            st.session_state["avisos_carga"] = []
            st.session_state["ia_resultado"] = None
            st.rerun()

    if archivo_subido is not None:
        firma_archivo = f"{archivo_subido.name}_{archivo_subido.size}"
        if st.session_state.get("firma_archivo_cargado") != firma_archivo:
            try:
                v_up, c_up, p_up, avisos = procesar_ventas_subidas(
                    archivo_subido,
                    st.session_state["df_clientes_base"],
                    st.session_state["df_productos_base"],
                )
                st.session_state["df_ventas_base"] = v_up
                st.session_state["df_clientes_base"] = c_up
                st.session_state["df_productos_base"] = p_up
                st.session_state["origen_datos"] = f"Archivo subido: {archivo_subido.name}"
                st.session_state["avisos_carga"] = avisos
                st.session_state["firma_archivo_cargado"] = firma_archivo
                st.session_state["ia_resultado"] = None
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

    st.info(f"**Fuente activa:** {st.session_state['origen_datos']}")
    for aviso in st.session_state.get("avisos_carga", []):
        st.caption(f"⚠️ {aviso}")

    # Preparar dataset enriquecido completo para poblar filtros
    df_completo = enriquecer_datos(
        st.session_state["df_ventas_base"],
        st.session_state["df_clientes_base"],
        st.session_state["df_productos_base"],
    )

    # 2. FILTROS INTERACTIVOS
    st.divider()
    st.subheader("2. Filtros Interactivos")

    fecha_min = df_completo["Fecha"].min().date()
    fecha_max = df_completo["Fecha"].max().date()

    rango_fechas = st.date_input(
        "Rango de fechas",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max,
    )

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        f_inicio, f_fin = rango_fechas
    elif isinstance(rango_fechas, (list, tuple)) and len(rango_fechas) == 1:
        f_inicio = f_fin = rango_fechas[0]
    else:
        f_inicio = f_fin = rango_fechas

    if f_inicio > f_fin:
        f_inicio, f_fin = f_fin, f_inicio

    ciudades_disp = sorted(df_completo["Ciudad"].dropna().unique().tolist())
    ciudades_sel = st.multiselect("Ciudad", options=ciudades_disp, default=ciudades_disp)

    df_suc_pool = df_completo[df_completo["Ciudad"].isin(ciudades_sel)] if ciudades_sel else df_completo
    sucursales_disp = sorted(df_suc_pool["Sucursal"].dropna().unique().tolist())
    sucursales_sel = st.multiselect("Sucursal", options=sucursales_disp, default=sucursales_disp)

    segmentos_disp = sorted(df_completo["Segmento"].dropna().unique().tolist())
    segmentos_sel = st.multiselect("Segmento de cliente", options=segmentos_disp, default=segmentos_disp)

    categorias_disp = sorted(df_completo["Categoria"].dropna().unique().tolist())
    categorias_sel = st.multiselect("Categoría", options=categorias_disp, default=categorias_disp)

    df_prod_pool = df_completo[df_completo["Categoria"].isin(categorias_sel)] if categorias_sel else df_completo
    productos_disp = sorted(df_prod_pool["Producto"].dropna().unique().tolist())
    productos_sel = st.multiselect("Producto", options=productos_disp, default=productos_disp)

    # 3. CONFIGURACIÓN OPCIONAL DE IA
    st.divider()
    st.subheader("3. Configuración de IA (Opcional)")
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-... (opcional; funciona sin clave)",
        help="Si se deja vacía, el asistente usará el motor de reglas comerciales automáticas.",
    )
    modelo_ia = st.selectbox("Modelo", options=["gpt-4o-mini", "gpt-4o"], index=0)


# ==============================================================================
# APLICACIÓN DE FILTROS AL DATASET
# ==============================================================================
mascara = (
    (df_completo["Fecha"].dt.date >= f_inicio)
    & (df_completo["Fecha"].dt.date <= f_fin)
    & (df_completo["Ciudad"].isin(ciudades_sel if ciudades_sel else ciudades_disp))
    & (df_completo["Sucursal"].isin(sucursales_sel if sucursales_sel else sucursales_disp))
    & (df_completo["Segmento"].isin(segmentos_sel if segmentos_sel else segmentos_disp))
    & (df_completo["Categoria"].isin(categorias_sel if categorias_sel else categorias_disp))
    & (df_completo["Producto"].isin(productos_sel if productos_sel else productos_disp))
)

df_filtrado = df_completo[mascara].copy()

# ==============================================================================
# CABECERA EJECUTIVA Y KPIs (st.metric)
# ==============================================================================
st.title("🍕 San Francisco Crust Pizza — Cuadro de Mando Ejecutivo")
st.markdown(
    f"Monitorización de ingresos, rentabilidad por producto y sucursal, comportamiento de segmentos y recomendaciones asistidas por IA · "
    f"**Periodo:** `{f_inicio}` a `{f_fin}`"
)

if df_filtrado.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados. Ajusta los criterios en la barra lateral.")
    st.stop()

ventas_totales = float(df_filtrado["Venta_Total_USD"].sum())
utilidad_bruta = float(df_filtrado["Utilidad_Bruta_USD"].sum())
margen_bruto_pct = (utilidad_bruta / ventas_totales * 100.0) if ventas_totales > 0 else 0.0
total_transacciones = int(df_filtrado["ID_Transaccion"].nunique())
ticket_promedio = (ventas_totales / total_transacciones) if total_transacciones > 0 else 0.0
clientes_unicos = int(df_filtrado["ID_Cliente"].nunique())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Ventas Totales ($)", f"${ventas_totales:,.2f}")
k2.metric("Utilidad Bruta ($)", f"${utilidad_bruta:,.2f}")
k3.metric("Margen Bruto (%)", f"{margen_bruto_pct:.1f}%")
k4.metric("Ticket Promedio ($)", f"${ticket_promedio:,.2f}")
k5.metric("Clientes Únicos", f"{clientes_unicos:,}")
k6.metric("Transacciones", f"{total_transacciones:,}")

st.divider()

# ==============================================================================
# GRÁFICAS INTERACTIVAS CON PLOTLY
# ==============================================================================
st.subheader("📊 Análisis Visual de Ventas y Rentabilidad")

# Fila 1: Top productos vendidos/rentables + Evolución temporal
col_g1, col_g2 = st.columns([1.25, 1.0])

with col_g1:
    metrica_prod = st.radio(
        "Ordenar y visualizar Top Productos por:",
        options=["Ventas Totales ($)", "Utilidad Bruta ($)", "Unidades Vendidas"],
        horizontal=True,
        key="radio_top_prod",
    )
    df_prod_agg = (
        df_filtrado.groupby(["Producto", "Categoria"], as_index=False)
        .agg(
            Ventas_USD=("Venta_Total_USD", "sum"),
            Utilidad_USD=("Utilidad_Bruta_USD", "sum"),
            Unidades=("Cantidad", "sum"),
        )
    )
    col_map_prod = {
        "Ventas Totales ($)": ("Ventas_USD", "Ventas ($)"),
        "Utilidad Bruta ($)": ("Utilidad_USD", "Utilidad Bruta ($)"),
        "Unidades Vendidas": ("Unidades", "Unidades"),
    }
    campo_x, titulo_x = col_map_prod[metrica_prod]
    df_top_prod = df_prod_agg.sort_values(campo_x, ascending=True).tail(8)

    fig_prod = px.bar(
        df_top_prod,
        x=campo_x,
        y="Producto",
        color="Categoria",
        orientation="h",
        title=f"Top Productos por {metrica_prod}",
        labels={campo_x: titulo_x, "Producto": "", "Categoria": "Categoría"},
        hover_data={"Ventas_USD": ":,.2f", "Utilidad_USD": ":,.2f", "Unidades": ":,", "Categoria": True},
        template="plotly_white",
    )
    fig_prod.update_layout(height=390, margin=dict(l=10, r=20, t=45, b=20), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_prod, use_container_width=True)

with col_g2:
    df_diario = (
        df_filtrado.groupby(df_filtrado["Fecha"].dt.to_period("D").dt.to_timestamp(), as_index=False)
        .agg(Ventas_USD=("Venta_Total_USD", "sum"), Utilidad_USD=("Utilidad_Bruta_USD", "sum"))
        .sort_values("Fecha")
    )
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=df_diario["Fecha"],
            y=df_diario["Ventas_USD"],
            mode="lines+markers",
            name="Ventas ($)",
            hovertemplate="%{x|%d %b %Y}<br>Ventas: $%{y:,.2f}<extra></extra>",
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=df_diario["Fecha"],
            y=df_diario["Utilidad_USD"],
            mode="lines",
            name="Utilidad Bruta ($)",
            hovertemplate="%{x|%d %b %Y}<br>Utilidad: $%{y:,.2f}<extra></extra>",
        )
    )
    fig_trend.update_layout(
        title="Evolución Diaria de Ventas y Utilidad Bruta",
        xaxis_title="Fecha",
        yaxis_title="USD ($)",
        template="plotly_white",
        height=390,
        margin=dict(l=10, r=20, t=45, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# Fila 2: Ventas por ciudad y sucursal + Ventas por segmento de cliente
col_g3, col_g4 = st.columns([1.15, 1.0])

with col_g3:
    df_suc_agg = (
        df_filtrado.groupby(["Ciudad", "Sucursal"], as_index=False)
        .agg(
            Ventas_USD=("Venta_Total_USD", "sum"),
            Utilidad_USD=("Utilidad_Bruta_USD", "sum"),
            Transacciones=("ID_Transaccion", "nunique"),
        )
        .sort_values("Ventas_USD", ascending=True)
    )
    fig_suc = px.bar(
        df_suc_agg,
        x="Ventas_USD",
        y="Sucursal",
        color="Ciudad",
        orientation="h",
        title="Ventas por Ciudad y Sucursal ($)",
        labels={"Ventas_USD": "Ventas Totales ($)", "Sucursal": "", "Ciudad": "Ciudad"},
        hover_data={"Utilidad_USD": ":,.2f", "Transacciones": ":,", "Ciudad": True},
        template="plotly_white",
    )
    fig_suc.update_layout(height=370, margin=dict(l=10, r=20, t=45, b=20), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_suc, use_container_width=True)

with col_g4:
    df_seg_agg = (
        df_filtrado.groupby("Segmento", as_index=False)
        .agg(
            Ventas_USD=("Venta_Total_USD", "sum"),
            Utilidad_USD=("Utilidad_Bruta_USD", "sum"),
            Clientes=("ID_Cliente", "nunique"),
        )
        .sort_values("Ventas_USD", ascending=False)
    )
    fig_seg = px.pie(
        df_seg_agg,
        names="Segmento",
        values="Ventas_USD",
        hole=0.45,
        title="Participación de Ventas por Segmento de Cliente",
        hover_data={"Utilidad_USD": ":,.2f", "Clientes": ":,"},
        template="plotly_white",
    )
    fig_seg.update_traces(textposition="inside", textinfo="percent+label")
    fig_seg.update_layout(height=370, margin=dict(l=10, r=20, t=45, b=20), showlegend=False)
    st.plotly_chart(fig_seg, use_container_width=True)

st.divider()

# ==============================================================================
# ASISTENTE DE IA COMERCIAL
# ==============================================================================
st.subheader("🤖 Asistente de IA Comercial")
st.markdown(
    "Analiza automáticamente los datos filtrados en pantalla para detectar oportunidades de margen, "
    "rendimiento por sucursal y acciones comerciales prioritarias."
)

col_ia_btn, col_ia_info = st.columns([1.2, 2.8])
with col_ia_btn:
    generar_btn = st.button(
        "Generar Análisis y Recomendaciones con IA",
        type="primary",
        use_container_width=True,
    )
with col_ia_info:
    estado_motor = "OpenAI conectado" if (api_key_input or "").strip() else "Modo Fallback (Reglas Comerciales Activo)"
    st.caption(f"Estado actual del motor: **{estado_motor}** · Analiza {len(df_filtrado):,} registros filtrados.")

if generar_btn:
    resumen_actual = construir_resumen_ejecutivo(df_filtrado)
    with st.spinner("Analizando métricas comerciales y generando recomendaciones..."):
        texto_ia, modo_usado = generar_analisis_ia(resumen_actual, api_key=api_key_input, modelo=modelo_ia)
        st.session_state["ia_resultado"] = texto_ia
        st.session_state["ia_modo"] = modo_usado

if st.session_state["ia_resultado"]:
    st.markdown(f"<div class='ai-card'>{st.session_state['ia_resultado']}</div>", unsafe_allow_html=True)
    st.caption(f"Motor utilizado: {st.session_state['ia_modo']}")

# ==============================================================================
# DETALLE DE TRANSACCIONES (DESPLEGABLE LIMPIO PARA DEMOSTRACIONES)
# ==============================================================================
with st.expander("📄 Ver detalle de transacciones filtradas", expanded=False):
    columnas_mostrar = [
        "ID_Transaccion",
        "Fecha",
        "Ciudad",
        "Sucursal",
        "Segmento",
        "Categoria",
        "Producto",
        "Tamaño",
        "Cantidad",
        "PrecioVentaUSD",
        "Venta_Total_USD",
        "Utilidad_Bruta_USD",
        "Metodo_Pago",
    ]
    cols_existentes = [c for c in columnas_mostrar if c in df_filtrado.columns]
    df_vista = df_filtrado[cols_existentes].sort_values("Fecha", ascending=False).head(300)
    st.dataframe(df_vista, use_container_width=True, hide_index=True)