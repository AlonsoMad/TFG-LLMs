"""Python wrapper around the Mallet Polylingual Topic Model implementation.
# https://mallet.cs.umass.edu/topics-polylingual.php

The input to the Mallet Polylingual Topic Model is a corpus of documents in multiple languages. This corpus of documents is given as a dataframe with the following columns:

- id_preproc: 
    Identifier used during the preprocessing of the document.
- lemmas:
    The lemmatized text of the document in the original language.
- lemmas_tr:
    The lemmatized text of the document in the target language.
- doc_id: 
    A unique identifier for each document, in the form "LANG_ID", where LANG is the language of the document and ID is a unique identifier. 
- text:
    The raw text of the document in the original language.
- text_tr:
    The raw text of the document in the target language.
- lang: The language of the document. 

The output is a folder with the following strucutre:
- model_folder
|--- train_data
|    |--- corpus_lang1.txt
|    |--- corpus_lang2.txt
|--- mallet_input
|    |--- corpus_lang1.mallet
|    |--- corpus_lang2.mallet
|--- mallet_output
|    |--- doc-topics.txt
|    |--- inferencer.mallet.0
|    |--- inferencer.mallet.1
|    |--- output-state.gz
|    |--- topickeys.txt
"""


import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os
import shutil
import numpy as np
from scipy import sparse
import gzip
import json

from sklearn.preprocessing import normalize

from src.utils.utils import file_lines


class PolylingualTM(object):

    def __init__(
        self,
        lang1: str,
        lang2: str,
        model_folder: str,
        num_topics: int,
        alpha: float = 1.0,
        token_regexp: str = r"\p{L}+",
        mallet_path: str = "src/topic_modeling/Mallet-202108/bin/mallet",
        add_stops_path: str = "src/topic_modeling/stops",
        is_second_level: bool = False,
        logger: logging.Logger = None
    ) -> None:
        """Initialize the PolylingualTM object.

        Parameters
        ----------
        lang1 : str
            The first language in the corpus.
        lang2 : str
            The second language in the corpus.
        model_folder : str
            The folder where all information related to the model will be stored.
        num_topics : int
            The number of topics to extract.
        alpha : float
            The alpha hyperparameter for the model.
        token_regexp : str
            The regular expression to use for tokenization.
        mallet_path : str
            The path to the Mallet executable.

        """

        self._lang1 = lang1
        self._lang2 = lang2
        self._num_topics = num_topics
        self._alpha = alpha
        self._token_regexp = token_regexp
        self._model_folder = model_folder
        self._mallet_path = pathlib.Path(mallet_path)
        self._add_stops_path = pathlib.Path(add_stops_path)
        self._is_second_level = is_second_level
        self._lang_lengths = {}
        self._docs_lang = {}
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('PolylingualTM')
            # Add a console handler to output logs to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Set handler level to INFO or lower
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Create folder for the model
        # If a model with the same name already exists, save a copy; then create a new folder
        if not self._is_second_level:
            if self._model_folder.exists():
                self._logger.info(
                    f"-- -- Given model folder {self._model_folder} already exists. Saving a copy ..."
                )
                old_model_folder = self._model_folder.parent / \
                    (self._model_folder.name + "_old")
                if not old_model_folder.is_dir():
                    os.makedirs(old_model_folder)
                    shutil.move(self._model_folder, old_model_folder)

            self._model_folder.mkdir(exist_ok=True)

        return

    def _create_mallet_input_corpus(
        self,
        df_path: pathlib.Path,
    ) -> int:
        """Given a corpus of documents, create the input files for the Mallet Polylingual Topic Model, i.e., creates 'corpus_lang1.txt' and 'corpus_lang2.txt' files.

        Parameters
        ----------
        df_path : pd.DataFrame
            Path to a dataframe with the following columns: `doc_id`, `lang`, and `raw_text`.

        Returns
        -------
        status : int
            Status of the operation:
            - 2 if the operation was successful.
            - 1 if the input format is incorrect.
            - 0 if the operation failed.
        """

        # Create 'train_data' folder
        self._train_data_folder = self._model_folder / "train_data"
        self._train_data_folder.mkdir(exist_ok=True)
        
        if not self._is_second_level:

            # Read the dataframe and create the input files
            df = pd.read_parquet(df_path)
            for lang in [self._lang1, self._lang2]:
                df_lang = df.copy()
                
                self._docs_lang[lang] = df_lang[df_lang["lang"] == lang]
                self._lang_lengths[lang] = len(self._docs_lang[lang])
                
                df_lang["lemmas"] = np.where(df_lang["lang"] != lang, df_lang["lemmas_tr"], df_lang["lemmas"])

                if df_lang.empty:
                    self._logger.error(
                        f"-- -- No documents found for language {lang}.")
                    return 1
                corpus_txt_path = self._train_data_folder / f"corpus_{lang}.txt"
                self._logger.info(
                    f"-- -- Creating Mallet {corpus_txt_path.as_posix()}...")
                with corpus_txt_path.open("w", encoding="utf8") as fout:
                    for i, t in zip(df_lang.doc_id, df_lang.lemmas):
                        fout.write(f"{i} {lang.upper()} {t}\n")
                self._logger.info(
                    f"-- -- Mallet {corpus_txt_path.as_posix()} created.")

        if (self._train_data_folder / f"corpus_{self._lang1}.txt").exists() and (self._train_data_folder / f"corpus_{self._lang2}.txt").exists():
            return 2
        else:
            return 0

    def _prepare_mallet_input(
        self,
    ) -> int:
        """Assuming there is a 'corpus_es.txt' and 'corpus_en.txt' files in the model folder, prepare the input files for the Mallet Polylingual Topic Model, i.e., generates the files 'corpus_lang1.mallet' and 'corpus_lang2.mallet'.

        Returns
        -------
        status : int
            Status of the operation:
            - 2 if the operation was successful.
            - 1 if the input files are missing.
            - 0 if the operation failed.
        """

        # Create folder for saving Mallet input files
        self._mallet_input_folder = self._model_folder / "mallet_input"
        self._mallet_input_folder.mkdir(exist_ok=True)

        # Transform training data into the format expected by Mallet
        self._logger.info(f"-- -- Importing data to Mallet...")
        for lang in [self._lang1, self._lang2]:
            corpus_txt_path = self._train_data_folder / f"corpus_{lang}.txt"
            corpus_mallet = self._mallet_input_folder / f"corpus_{lang}.mallet"
            
            stw_file = self._add_stops_path / f"{lang.lower()}.txt"

            cmd = self._mallet_path.as_posix() + \
                ' import-file --preserve-case --keep-sequence ' + \
                '--remove-stopwords --token-regex "' + self._token_regexp + \
                '" --print-output --input %s --output %s --extra-stopwords %s'
            cmd = cmd % (corpus_txt_path, corpus_mallet, stw_file)

            try:
                self._logger.info(f'-- -- Running command {cmd}')
                check_output(args=cmd, shell=True)
            except:
                self._logger.error(
                    '-- -- Mallet failed to import data. Revise command')
                return 0

            self._logger.info(f"-- -- Data imported to Mallet.")

        if (self._mallet_input_folder / "corpus_lang1.mallet").exists() and (self._mallet_input_folder / "corpus_lang2.mallet").exists():
            return 2
        else:
            return 1

    def train(
        self,
        df_path: pathlib.Path=None,
    ) -> int:
        """Assuming there is a 'corpus_es.mallet' and 'corpus_en.mallet' files in the model folder, train the Mallet Polylingual Topic Model.

        Returns
        -------
        status : int
            Status of the operation:
            - 2 if the operation was successful.
            - 1 if the input files are missing.
            - 0 if the operation failed.
        """

        # Create the input files for Mallet
        self._create_mallet_input_corpus(df_path)

        # Prepare the input files for Mallet
        self._prepare_mallet_input()

        # Create folder for saving Mallet output
        self._mallet_out_folder = self._model_folder / "mallet_output"
        self._mallet_out_folder.mkdir(exist_ok=True)

        # Actual training of the model
        mallet_input_files = [self._mallet_input_folder /  f"corpus_{lang}.mallet" for lang in [self._lang1, self._lang2]]
        language_inputs = ' '.join(
            [file.resolve().as_posix() for file in mallet_input_files])
        cmd = self._mallet_path.as_posix() + \
            ' run cc.mallet.topics.PolylingualTopicModel  ' + \
            '--language-inputs %s --num-topics %s --alpha %s ' + \
            '--output-state %s '\
            '--output-doc-topics %s '\
            '--output-topic-keys %s '\
            '--inferencer-filename %s '
        cmd = cmd % \
            (language_inputs, self._num_topics, self._alpha,
             (self._mallet_out_folder / "output-state.gz").resolve().as_posix(),
             (self._mallet_out_folder / "doc-topics.txt").resolve().as_posix(),
             (self._mallet_out_folder / "topickeys.txt").resolve().as_posix(),
             (self._mallet_out_folder / "inferencer.mallet").resolve().as_posix(),
             )

        try:
            self._logger.info(
                f'-- -- Training Mallet PolylingualTopicModel. Command is {cmd}')
            check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Model training failed. Revise command')
            return
        
        self._logger.info(f"-- -- Saving model information...")
        self.save_model_info()
        self._logger.info(f"-- -- Model information saved successfully.")
        
        return
        
    def save_model_info(self):
        
        tuples_lang = [
            (self._lang1, 0),
            (self._lang2, 1)
        ]
        
        ########################################################################
        # THETAS
        ########################################################################
        self._logger.info(f"-- -- Getting thetas...")
        # Define file paths
        thetas_file = self._mallet_out_folder / "doc-topics.txt"

        # Get number of documents
        lang1_nr_docs = self._lang_lengths[self._lang1]
        lang2_nr_docs = self._lang_lengths[self._lang2]

        # Initialize theta matrices
        lang1_thetas = np.zeros((lang1_nr_docs, self._num_topics))
        lang2_thetas = np.zeros((lang2_nr_docs, self._num_topics))

        # Read and parse the thetas file
        with open(thetas_file, 'r') as file:
            lines = file.readlines()[1:]  # Skip the first line

        for line in lines:
            values = line.split()
            doc_id = int(values[0])
            for tpc in range(1, len(values), 2):
                topic_id = int(values[tpc])
                weight = float(values[tpc + 1])
                if doc_id < lang1_nr_docs:
                    lang1_thetas[doc_id, topic_id] = weight if not np.isnan(weight) else 0
                else:
                    lang2_thetas[doc_id - lang1_nr_docs, topic_id] = weight if not np.isnan(weight) else 0

        # Normalize the thetas
        lang1_thetas = normalize(lang1_thetas, axis=1, norm='l1')
        lang1_thetas = sparse.csr_matrix(lang1_thetas, copy=True)

        lang2_thetas = normalize(lang2_thetas, axis=1, norm='l1')
        lang2_thetas = sparse.csr_matrix(lang2_thetas, copy=True)

        # Calculate mean topic proportions (alphas)
        lang1_alphas = np.asarray(np.mean(lang1_thetas, axis=0)).ravel()
        lang2_alphas = np.asarray(np.mean(lang2_thetas, axis=0)).ravel()

        # Save the thetas and alphas
        sparse.save_npz(self._mallet_out_folder / f"thetas_{self._lang1}.npz", lang1_thetas)
        sparse.save_npz(self._mallet_out_folder / f"thetas_{self._lang2}.npz", lang2_thetas)
        np.save(self._mallet_out_folder / f"alphas_{self._lang1}.npy", lang1_alphas)
        np.save(self._mallet_out_folder / f"alphas_{self._lang2}.npy", lang2_alphas)

        thetas_dict = {
            self._lang1: lang1_thetas,
            self._lang2: lang2_thetas
        }

        
        ########################################################################
        # VOCABS
        ########################################################################
        # 0 = document's id
        # 1 = lang (0 for en, 1 for es)
        # 3 = id of the word in the document
        # 4 = id of the word in the vocabulary
        # 5 = word
        # 6 = topic to which the word belongs
        self._logger.info(f"-- -- Getting vocab...")
        topic_state_model = self._mallet_out_folder / "output-state.gz"
        with gzip.open(topic_state_model) as fin:
            topic_state_df = pd.read_csv(
                fin, delim_whitespace=True,
                names=['docid', 'lang', 'wd_docid','wd_vocabid', 'wd', 'tpc'],
                header=None, skiprows=1)
        """
        betas_dict = {}
        for lang, id_lang in tuples_lang:
            # Filter by lang
            df_lang = topic_state_df[topic_state_df.lang == id_lang]
            
            vocab_size = len(df_lang.wd_vocabid.unique())
            num_topics = len(df_lang.tpc.unique())
            betas = np.zeros((num_topics, vocab_size))
            vocab = list(df_lang.wd.unique())
            term_freq = np.zeros((vocab_size,))
            
            # Group by 'tpc' and 'wd_vocabid', and count occurrences
            grouped = df_lang.groupby(['tpc', 'wd_vocabid']).size().reset_index(name='count')

            # Populate the betas matrix with the counts
            for _, row in grouped.iterrows():
                tpc = row['tpc']
                vocab_id = row['wd_vocabid']
                count = row['count']
                betas[tpc, vocab_id] = count
                term_freq[vocab_id] += count
            betas = normalize(betas, axis=1, norm='l1')
            
            betas_dict[lang] = betas
                
            vocab_file = self._mallet_out_folder / f"vocab_{lang}.txt"
            betas_file = self._mallet_out_folder / f"betas_{lang}.npy"
            with vocab_file.open('w', encoding='utf8') as fout:
                for el in zip(vocab, term_freq):
                    term = str(el[0])
                    try:
                        freq = int(float(el[1]))  # Convert to float first, then int
                        fout.write(f"{term}\t{freq}\n")
                    except (ValueError, TypeError):
                        print(f"Skipping invalid term_freq value: {el[1]}")
            np.save(betas_file, betas)
        """
        # Shared vocabulary across all languages
        vocab_size = len(topic_state_df.wd_vocabid.unique())
        num_topics = len(topic_state_df.tpc.unique())
        betas = np.zeros((num_topics, vocab_size))
        vocab = list(topic_state_df.wd.unique())
        term_freq = np.zeros((vocab_size,))

        # Group by 'tpc' and 'wd_vocabid' to count occurrences
        grouped = topic_state_df.groupby(['tpc', 'wd_vocabid']).size().reset_index(name='count')

        # Populate the betas matrix with counts
        for _, row in grouped.iterrows():
            tpc = row['tpc']
            vocab_id = row['wd_vocabid']
            count = row['count']
            betas[tpc, vocab_id] = count
            term_freq[vocab_id] += count

        # Normalize betas
        betas = normalize(betas, axis=1, norm='l1')

        # Save shared vocab and betas
        vocab_file = self._mallet_out_folder / "vocab.txt"
        betas_file = self._mallet_out_folder / "betas.npy"
        with vocab_file.open('w', encoding='utf8') as fout:
            for el in zip(vocab, term_freq):
                term = str(el[0])
                try:
                    freq = int(float(el[1]))
                    fout.write(f"{term}\t{freq}\n")
                except (ValueError, TypeError):
                    print(f"Skipping invalid term_freq value: {el[1]}")
        np.save(betas_file, betas)

        ########################################################################
        # KEYS
        ########################################################################
        # one line for topic id + alpha
        # then out.print(" " + language + "\t" + languageTokensPerTopic[language][topic] + "\t" + betas[language] + "\t"); words
        self._logger.info(f"-- -- Getting keys...")
        topic_keys = self._mallet_out_folder / "topickeys.txt"
        topic_keys_df = pd.read_csv(
            topic_keys, delimiter="\t", 
            names = ['lang', 'langTokensPerTopic', 'betas', 'topK'],
            header=None
        ).dropna()
        
        # Save keys in different files for each language
        for lang, id_lang in tuples_lang:
            # Filter by lang
            df_lang = topic_keys_df[topic_keys_df.lang == id_lang]
            
            keys_file = self._mallet_out_folder / f"keys_{lang}.txt"
            with open(keys_file, 'w') as file:
                # Iterate over each value in the column and write it to the file
                for value in df_lang["topK"]:
                    file.write(str(value) + '\n')
            
        ########################################################################
        # S3
        ########################################################################
        for lang, id_lang in tuples_lang:
            # Get the thetas for the current language
            thetas = thetas_dict[lang].toarray()

            # Load the shared betas (no longer language-specific)
            betas_file = self._mallet_out_folder / "betas.npy"
            betas = np.load(betas_file)

            # Load the vocab for mapping words to IDs
            vocab_w2id = {}
            vocab_id2w = {}
            vocab_file = self._mallet_out_folder / "vocab.txt"
            with open(vocab_file) as file:
                for i, line in enumerate(file):
                    # Strip leading and trailing whitespace
                    stripped_line = line.strip()
                    # Split the line into words and numbers
                    parts = stripped_line.split()
                    if parts:
                        # Get the word (first part)
                        wd = parts[0]
                        # Populate the dictionaries
                        vocab_w2id[wd] = i
                        vocab_id2w[str(i)] = wd

            # Prepare to compute S3
            documents_texts = self._docs_lang[lang]["lemmas"].apply(lambda x: x.split()).tolist()
            D = len(thetas)  # Number of documents
            K = betas.shape[0]  # Number of topics (shared across languages)
            S3 = np.zeros((D, K))

            # Compute S3 by summing beta probabilities for the document's words
            for doc in range(D):
                for topic in range(K):
                    # Get word IDs for the current document
                    wd_ids = [vocab_w2id[word] for word in documents_texts[doc] if word in vocab_w2id]
                    # Sum beta values for the document's words under the current topic
                    S3[doc, topic] = np.sum(betas[topic, wd_ids])

            # Convert S3 to a sparse matrix and save
            S3 = sparse.csr_matrix(S3, copy=True)
            s3_file = self._mallet_out_folder / f"s3_{lang}.npz"
            sparse.save_npz(s3_file, S3)


        return