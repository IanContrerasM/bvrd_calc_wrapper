import typing
from typing import Dict, List

import requests

if typing.TYPE_CHECKING:
    from loguru import Logger
import math

import numpy as np
import pandas as pd


class BVRDCalculator:
    """
    Wrapper class for BVRD calculator API that provides simplified methods for
    valuing financial instruments in the Dominican market.
    """

    BASE_URL = "https://calculadora.testinnex.exchange"
    MAX_ROWS_PER_REQUEST = 20_000

    def __init__(self, username: str, password: str, logger: "Logger") -> None:
        """
        Initialize the BVRD calculator client.

        Args:
            username: API username
            password: API password
        """
        self.username = username
        self.password = password
        self.logger = logger

    def _call_api(self, endpoint: str, payload: Dict) -> Dict:
        """
        Make an API call to the BVRD calculator.

        Args:
            endpoint: API endpoint (e.g., "/apicbbvrd")
            payload: Request payload

        Returns:
            API response as dictionary
        """
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API call failed: {str(e)}")
            raise e

    def _make_request_body(self, calc_body: List[Dict], config: Dict) -> Dict:
        """
        Create the request body for the API call.

        Args:
            calc_body: Calculation body

        Returns:
            Request body as dictionary
        """
        return {
            "auth": {
                "usuario": self.username,
                "password": self.password,
            },
            "calculo": calc_body,
            "config": config,
        }


class BondCalculator(BVRDCalculator):
    def _make_calc_body(
        self,
        isin: pd.Series | float,
        input_type: str,
        amount_type: str,
        input: pd.Series | float,
        amount: pd.Series | float,
        date: pd.Series,
        id_calculo: pd.Series | int | None = None,
    ):
        data = {
            "titulo_id": isin,
            "tipo_insumo": input_type,
            "tipo_monto": amount_type,
            "insumo": input,
            "monto": amount,
            "fecha_liquidacion": date,
        }

        if id_calculo is not None:
            data["id_calculo"] = id_calculo

        return pd.DataFrame(data)

    def _unpack_response(self, response: Dict):
        """
        Unpack API response into two DataFrames: valuation and cashflows.

        Args:
            response: API response as a dictionary.

        Returns:
            A tuple of (valuation_df, cashflows_df)
        """
        valuations = []
        cashflows = []

        for item in response:
            # Si existe la llave 'titulo_calculo', estamos en el escenario con cashflow.
            if "titulo_calculo" in item:
                titulo_calculo = item.get("titulo_calculo", {})
                valuations.append(titulo_calculo)
                flujos_titulo = item.get("flujos_titulo", [])
                # Si hay flujos, se procesan y se asocia el ISIN para trazabilidad.
                if flujos_titulo:
                    for flujo in flujos_titulo:
                        flujo["id_calculo"] = titulo_calculo.get("id_calculo")
                        cashflows.append(flujo)
            else:
                # Caso sin cashflow: la respuesta contiene directamente los datos de valoración.
                valuations.append(item)

        valuation_df = pd.DataFrame(valuations)
        # Si no hay cashflows, se retorna un DataFrame vacío para esa parte.
        cashflows_df = pd.DataFrame(cashflows) if cashflows else pd.DataFrame()

        return valuation_df, cashflows_df

    def NPV(
        self,
        isin: pd.Series | float,
        input_type: str,
        amount_type: str,
        input: pd.Series | float,
        amount: pd.Series | float,
        date: pd.Series,
        id_calculo: pd.Series | int,
        with_cashflow: bool = False,
    ):
        df = self._make_calc_body(
            isin, input_type, amount_type, input, amount, date, id_calculo
        )
        total_rows = len(df)

        if total_rows == 0:
            self.logger.warning("Empty input for NPV calculation.")
            return pd.DataFrame(), pd.DataFrame()

        self.logger.info(
            f"Processing {total_rows} rows in chunks of {self.MAX_ROWS_PER_REQUEST}"
        )

        valuation_chunks = []
        cashflow_chunks = []

        num_chunks = math.ceil(total_rows / self.MAX_ROWS_PER_REQUEST)

        for i in range(num_chunks):
            start = i * self.MAX_ROWS_PER_REQUEST
            end = start + self.MAX_ROWS_PER_REQUEST
            chunk_df = df.iloc[start:end]

            request_body = self._make_request_body(
                chunk_df.to_dict(orient="records"),
                config={"with_flujos": with_cashflow},
            )

            self.logger.debug(
                f"Sending chunk {i + 1}/{num_chunks} with {len(chunk_df)} rows"
            )

            try:
                response = self._call_api("/apicbbvrd", request_body)
                valuation, cashflows = self._unpack_response(response)
                valuation_chunks.append(valuation)
                if not cashflows.empty:
                    cashflow_chunks.append(cashflows)
            except Exception as e:
                self.logger.error(f"Chunk {i + 1} failed: {str(e)}")
                raise

        final_valuation_df = pd.concat(valuation_chunks, ignore_index=True)
        final_cashflows_df = (
            pd.concat(cashflow_chunks, ignore_index=True)
            if cashflow_chunks
            else pd.DataFrame()
        )

        return final_valuation_df, final_cashflows_df

    def current_yield(self, valuation_df) -> pd.DataFrame:
        return valuation_df["cupon"] / valuation_df["precio_sucio"].replace(0, np.nan)

    def dollar_duration(self, valuation_df) -> pd.DataFrame:
        return valuation_df["precio_limpio"] * valuation_df["modified_duration"]

    def dollar_convexity(self, valuation_df) -> pd.DataFrame:
        return valuation_df["precio_limpio"] * valuation_df["convexidad"]

    def duration_to_convexity(self, valuation_df) -> pd.DataFrame:
        return valuation_df["modified_duration"] / valuation_df["convexidad"].replace(
            0, np.nan
        )

    def add_coupon_rate(self, valuation_df, cashflows_df):
        """
        Adds the 'tasa_interes' column to valuation_df by merging it with cashflows_df.

        Parameters:
            valuation_df (pd.DataFrame): The valuation DataFrame.
            cashflows_df (pd.DataFrame): The cashflows DataFrame.

        Returns:
            pd.DataFrame: The updated valuation_df with the 'tasa_interes' column.
        """
        # Perform the merge
        merged_df = valuation_df.merge(
            cashflows_df[["id_calculo", "fecha_flujo_str", "tasa_interes"]],
            how="left",
            left_on=["id_calculo", "fecha_liquidacion_str"],
            right_on=["id_calculo", "fecha_flujo_str"],
        )
        return merged_df


class SBBCalculator(BVRDCalculator):
    """
    Calculator for Structured Bond Valuation using the updated BVRD API payload.
    """

    MAX_ROWS_PER_REQUEST = 100  # Set as needed

    def _make_calc_body(
        self,
        titulo_id: pd.Series | str,
        monto_transado_fwd: pd.Series | float,
        monto_transado_spot: pd.Series | float,
        monto_nominal: pd.Series | float,
        fecha_liquidacion_fwd: pd.Series | str,
        fecha_liquidacion_spot: pd.Series | str,
        base_dias: int = 360,
        id_calculo: pd.Series | int | None = None,
    ) -> pd.DataFrame:
        data = {
            "titulo_id": titulo_id,
            "monto_transado_fwd": monto_transado_fwd,
            "monto_transado_spot": monto_transado_spot,
            "monto_nominal": monto_nominal,
            "fecha_liquidacion_fwd": fecha_liquidacion_fwd,
            "fecha_liquidacion_spot": fecha_liquidacion_spot,
            "base_dias": base_dias,
        }

        if id_calculo is not None:
            data["id_calculo"] = id_calculo

        return pd.DataFrame(data)

    def _unpack_response(
        self, response: list[Dict]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        valuation = pd.DataFrame([item["calculo_estructurado"] for item in response])
        flujos = pd.DataFrame(
            [
                flujo
                for item in response
                for flujo in item.get("flujos_estructurado", [])
            ]
        )
        return valuation, flujos

    def NPV(
        self,
        titulo_id: pd.Series | str,
        monto_transado_fwd: pd.Series | float,
        monto_transado_spot: pd.Series | float,
        monto_nominal: pd.Series | float,
        fecha_liquidacion_fwd: pd.Series | str,
        fecha_liquidacion_spot: pd.Series | str,
        base_dias: int = 360,
        id_calculo: pd.Series | int | None = None,
        with_flujos: bool = False,
        round_precision: int = 6,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self._make_calc_body(
            titulo_id,
            monto_transado_fwd,
            monto_transado_spot,
            monto_nominal,
            fecha_liquidacion_fwd,
            fecha_liquidacion_spot,
            base_dias,
            id_calculo,
        )

        total_rows = len(df)
        if total_rows == 0:
            self.logger.warning("Empty input for NPV calculation.")
            return pd.DataFrame(), pd.DataFrame()

        self.logger.info(
            f"Processing {total_rows} rows in chunks of {self.MAX_ROWS_PER_REQUEST}"
        )

        valuation_chunks = []
        cashflow_chunks = []

        num_chunks = math.ceil(total_rows / self.MAX_ROWS_PER_REQUEST)

        for i in range(num_chunks):
            start = i * self.MAX_ROWS_PER_REQUEST
            end = start + self.MAX_ROWS_PER_REQUEST
            chunk_df = df.iloc[start:end]

            request_body = {
                "calculo": chunk_df.to_dict(orient="records"),
                "config": {
                    "with_flujos": with_flujos,
                    "round": round_precision,
                },
            }

            self.logger.debug(
                f"Sending chunk {i + 1}/{num_chunks} with {len(chunk_df)} rows"
            )

            try:
                response = self._call_api("/apicbbvrd_estructurado_rwd", request_body)
                valuation, flujos = self._unpack_response(response)
                valuation_chunks.append(valuation)
                if not flujos.empty:
                    cashflow_chunks.append(flujos)
            except Exception as e:
                self.logger.error(f"Chunk {i + 1} failed: {str(e)}")
                raise

        final_valuation_df = pd.concat(valuation_chunks, ignore_index=True)
        final_cashflows_df = (
            pd.concat(cashflow_chunks, ignore_index=True)
            if cashflow_chunks
            else pd.DataFrame()
        )

        return final_valuation_df, final_cashflows_df
