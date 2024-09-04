from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel

class DatabaseEmbedder:
    """
    Embeds database tables into a semantic space using BERT.
    Provides methods to find similar tables based on embeddings and word matching.

    Attributes:
        tokenizer (transformers.BertTokenizer): Tokenizer for converting text to tokens.
        model (transformers.BertModel): BERT model for computing embeddings.
        map_name_to_embeddings (dict): Mapping from table names to their embeddings.
        map_name_to_table_def (dict): Mapping from table names to their definitions.
    """

    def __init__(self):
        """Initializes the DatabaseEmbedder with pre-trained BERT tokenizer and model."""

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.map_name_to_embeddings = {}
        self.map_name_to_table_def = {}

    def add_table(self, table_name: str, text_representation: str):
        """
        Adds a table to the database, computing its embedding and storing its definition.

        Args:
            table_name (str): The name of the table.
            text_representation (str): A text representation of the table's schema or content.
        """

        self.map_name_to_embeddings[table_name] = self.compute_embeddings(text_representation)
        self.map_name_to_table_def[table_name] = text_representation
        print("\n----------- ADDING TABLES ------------------")
        print(f"{table_name}: {text_representation}")

    def compute_embeddings(self, text):
        """Computes the embedding for a given text using BERT.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The computed embedding.
        """

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        outputs = self.model(**inputs)
        return outputs["pooler_output"].detach().numpy()

    def get_similar_tables_via_embeddings(self, query, n=3):
        """
        Finds the top 'n' tables similar to a given query based on their embeddings.

        Args:
            query (str): The user's natural language query.
            n (int, optional): Number of top tables returned. Defaults to 3.

        Returns:
            list[str]: Top 'n' table names ranked by their similarity to the query.
        """

        query_embedding = self.compute_embeddings(query)
        print(f"QUERY IS: {query}")
        for name, emb in self.map_name_to_embeddings.items():
            print(name)

        similarities = {
            table: cosine_similarity(query_embedding, emb)[0][0]
            for table, emb in self.map_name_to_embeddings.items()
        }
        
        # Rank tables based on their similarity score, return the top 'n'
        print("\n---------------- EMBEDDING SIMILARITY ---------------")
        print(sorted(similarities, key=similarities.get, reverse=True)[:n])
        return sorted(similarities, key=similarities.get, reverse=True)[:n]

    def get_similar_table_via_word_match(self, query: str):
        """Finds tables that contain the query terms in their names.

        Args:
            query (str): The user's natural language query.

        Returns:
            list[str]: Table names that contain the query terms.
        """

        tables = []

        for table_name in self.map_name_to_table_def.keys():
            if table_name.lower() in query.lower():
                tables.append(table_name)

        print("\n---------------- QUERY SIMILARITY ---------------")
        print(tables)
        return tables

    def get_similar_tables(self, query: str, n=3):
        """
        Combines the results from `get_similar_tables_via_embeddings` and
        `get_similar_table_via_word_match`.

        Args:
            query (str): The user's natural language query.
            n (int, optional): Number of top tables returned. Defaults to 3.

        Returns:
            list[str]: Unique table names that are similar to the query.
        """

        similar_tables_via_embeddings = self.get_similar_tables_via_embeddings(query, n)
        similar_tables_via_word_match = self.get_similar_table_via_word_match(query)

        result = similar_tables_via_embeddings + similar_tables_via_word_match

        unique_results = []
        for item in result:
            if item not in unique_results:
                unique_results.append(item)

        return unique_results

    def get_table_definitions_from_names(self, table_names: list) -> list:
        """Retrieves the definitions for a list of table names.

        Args:
            table_names (list[str]): A list of table names.

        Returns:
            str: A string containing the definitions of the tables, separated by newline characters.
        """

        table_defs = [
            self.map_name_to_table_def[table_name] for table_name in table_names
        ]
        return "\n\n".join(table_defs)

