import pandas as pd
from pathlib import Path


# Colunas que o pipeline de preprocessing espera receber.
# Qualquer fonte de dados (CSV, banco, API) deve entregar exatamente essas colunas.
EXPECTED_COLUMNS: set[str] = {
    "Customer_ID",
    "Age",
    "Gender",
    "Annual_Income",
    "Total_Spend",
    "Years_as_Customer",
    "Num_of_Purchases",
    "Average_Transaction_Amount",
    "Num_of_Returns",
    "Num_of_Support_Contacts",
    "Satisfaction_Score",
    "Last_Purchase_Days_Ago",
    "Email_Opt_In",
    "Promotion_Response",
    "Target_Churn",
}


class ExtractionError(Exception):
    """Erro específico de extração — facilita o catch seletivo no pipeline."""
    pass


class DataExtractor:
    """
    Camada de extração: lê dados brutos de uma fonte e entrega um DataFrame
    validado, pronto para entrar no preprocessing.

    Responsabilidades desta classe:
    - Ler o arquivo (CSV por enquanto, extensível para banco/API).
    - Verificar se as colunas esperadas estão presentes.
    - Garantir tipos mínimos (Customer_ID como int, Target_Churn como bool).
    - NÃO faz limpeza, imputação, encoding nem feature engineering.
      Essas responsabilidades pertencem ao preprocessing/.

    Por que separar extração de preprocessing?
    - Em produção, a fonte muda (CSV → banco → stream), mas o preprocessing
      não deve saber disso. O extractor é o único ponto a trocar.
    - Erros de leitura (arquivo não encontrado, coluna faltando) devem
      explodir aqui, com mensagens claras, antes de entrar no pipeline.
    """

    def __init__(self, source: str | Path):
        """
        Parâmetros
        ----------
        source : str | Path
            Caminho para o arquivo CSV de entrada.
            Futuramente pode ser uma connection string ou URL.
        """
        self.source = Path(source)

    # ----------------------------
    # INTERFACE PÚBLICA
    # ----------------------------
    def extract(self) -> pd.DataFrame:
        """
        Lê, valida e retorna o DataFrame bruto.

        Raises
        ------
        ExtractionError
            Se o arquivo não for encontrado, estiver vazio ou
            tiver colunas obrigatórias ausentes.
        """
        df = self._read()
        self._validate_not_empty(df)
        self._validate_columns(df)
        df = self._cast_types(df)
        return df

    # ----------------------------
    # LEITURA
    # ----------------------------
    def _read(self) -> pd.DataFrame:
        if not self.source.exists():
            raise ExtractionError(
                f"Arquivo não encontrado: {self.source}. "
                "Verifique o caminho e tente novamente."
            )

        try:
            return pd.read_csv(self.source)
        except Exception as e:
            raise ExtractionError(f"Falha ao ler {self.source}: {e}") from e

    # ----------------------------
    # VALIDAÇÕES
    # ----------------------------
    def _validate_not_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ExtractionError(
                f"O arquivo {self.source} foi lido mas está vazio (0 linhas)."
            )

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ExtractionError(
                f"Colunas obrigatórias ausentes no dataset: {sorted(missing)}. "
                "Verifique se a fonte de dados está no formato correto."
            )

    # ----------------------------
    # TIPAGEM MÍNIMA
    # ----------------------------
    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Garante tipos mínimos antes de qualquer processamento.
        Não imputa nem transforma — só converte o que já existe.
        """
        df = df.copy()

        if "Customer_ID" in df.columns:
            df["Customer_ID"] = pd.to_numeric(df["Customer_ID"], errors="coerce").astype("Int64")

        if "Target_Churn" in df.columns:
            df["Target_Churn"] = df["Target_Churn"].astype(bool)

        if "Email_Opt_In" in df.columns:
            df["Email_Opt_In"] = df["Email_Opt_In"].astype(bool)

        return df