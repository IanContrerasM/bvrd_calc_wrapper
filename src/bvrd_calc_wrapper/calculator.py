import typing
from typing import Dict, List

import requests

if typing.TYPE_CHECKING:
    from loguru import Logger
import pandas as pd


class BVRDCalculator:
    """
    Wrapper class for BVRD calculator API that provides simplified methods for
    valuing financial instruments in the Dominican market.
    """

    BASE_URL = "https://calculadora.testinnex.exchange"

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
    ):
        df = pd.DataFrame(
            {
                "titulo_id": isin,
                "tipo_insumo": input_type,
                "tipo_monto": amount_type,
                "insumo": input,
                "monto": amount,
                "fecha_liquidacion": date,
            }
        )
        return df.to_dict(orient="records")

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
                        flujo["codisin"] = titulo_calculo.get("codisin")
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
        with_cashflow: bool = False,
    ):
        calc_body = self._make_calc_body(
            isin, input_type, amount_type, input, amount, date
        )
        request_body = self._make_request_body(
            calc_body, config={"with_flujos": with_cashflow}
        )
        response = self._call_api("/apicbbvrd", request_body)
        valuation, cashflows = self._unpack_response(response)
        return valuation, cashflows


class SBBCalculator(BVRDCalculator):
    """
    Calculator for Structured Bond Valuation using BVRD API.
    """

    def _make_calc_body(
        self,
        isin: pd.Series | str,
        input_type: str,
        amount_type: str,
        input: pd.Series | float,
        amount: pd.Series | float,
        date: pd.Series | str,
        days: pd.Series | int,
        cesion_cupon: bool = True,
        base_dias: int = 360,
    ) -> List[Dict]:
        df = pd.DataFrame(
            {
                "titulo_id": isin,
                "tipo_insumo": input_type,
                "tipo_monto": amount_type,
                "insumo": input,
                "monto": amount,
                "fecha_liquidacion_spot": date,
                "dias": days,
                "cesion_cupon": cesion_cupon,
                "base_dias": base_dias,
            }
        )

        # Calculate "tasa" column using the formula provided
        df["tasa"] = ((df["insumo"] - df["monto"]) / df["monto"]) * (
            base_dias / df["dias"]
        )

        return df.to_dict(orient="records")

    def _unpack_response(self, response: Dict) -> pd.DataFrame:
        return pd.DataFrame(response)

    def NPV(
        self,
        isin: pd.Series | str,
        input_type: str,
        amount_type: str,
        input: pd.Series | float,
        amount: pd.Series | float,
        date: pd.Series | str,
        days: pd.Series | int,
        cesion_cupon: bool = True,
        base_dias: int = 360,
    ) -> pd.DataFrame:
        calc_body = self._make_calc_body(
            isin,
            input_type,
            amount_type,
            input,
            amount,
            date,
            days,
            cesion_cupon,
            base_dias,
        )
        request_body = self._make_request_body(calc_body, config={})
        response = self._call_api("/apicbbvrd_estructurado", request_body)
        return self._unpack_response(response)
