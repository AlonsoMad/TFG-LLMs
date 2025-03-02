"""Python wrapper for the LDA Mallet Topic Model to train a topic model for each language in a corpus of documents.

The input to the LDA Mallet Topic Model is a corpus of documents in the same format as for the PolylingualTM wrapper, but each language is treated as a separate corpus to be modeled independently. The output is a folder with the following structure:

- model_folder
|--- train_data
|    |--- corpus_lang1.txt
|    |--- corpus_lang2.txt
|--- mallet_input
|    |--- corpus_lang1.mallet
|    |--- corpus_lang2.mallet
|--- mallet_output_lang1
|    |--- doc-topics.txt
|--- mallet_output_lang2
|    |--- doc-topics.txt
"""

import logging
import pathlib
from subprocess import check_output
import pandas as pd
import os
import shutil
import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse
from src.utils.utils import file_lines
import warnings

class LDATM(object):
    def __init__(
        self,
        lang1: str,
        lang2: str,
        model_folder: str,
        num_topics: int = 35,
        alpha: float = 5.0,
        optimize_interval: int = 10,
        num_threads: int = 4,
        num_iterations: int = 1000,
        doc_topic_thr: float = 0.0,
        token_regexp: str = "[\p{L}\p{N}][\p{L}\p{N}\p{P}]*\p{L}",
        thetas_thr=0.003,
        mallet_path: str = "src/topic_modeling/Mallet-202108/bin/mallet",
        logger: logging.Logger = None
    ) -> None:
        """Initialize the LDATM object.

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
        optimize_interval: int
            The interval for optimizing hyperparameters.
        num_threads: int
            The number of threads to use.
        num_iterations: int
            The number of iterations to run.
        doc_topic_thr: float
            The threshold for the document-topic distribution.
        token_regexp : str
            The regular expression to use for tokenization.
        mallet_path : str
            The path to the Mallet executable.

        """
        self._lang1 = lang1
        self._lang2 = lang2
        self._num_topics = num_topics
        self._alpha = alpha
        self._optimize_interval = optimize_interval
        self._num_threads = num_threads
        self._num_iterations = num_iterations
        self._doc_topic_thr = doc_topic_thr
        self._token_regexp = token_regexp
        self._thetas_thr = thetas_thr
        self._model_folder = model_folder
        self._mallet_path = pathlib.Path(mallet_path)

        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('PolylingualTM')
            # Add a console handler to output logs to the console
            console_handler = logging.StreamHandler()
            # Set handler level to INFO or lower
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Create folder for the model
        # If a model with the same name already exists, save a copy; then create a new folder
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
    ) -> None:
        """
        Given a corpus of documents, create the input files for Mallet. Each language is treated as a separate corpus, i.e., creates 'corpus_lang1.txt' and 'corpus_lang2.txt' files.

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

        # Read the dataframe and create the input files
        df = pd.read_parquet(df_path)
        for lang in [self._lang1, self._lang2]:
            df_lang = df.copy()
            # Filter the dataframe by language
            df_lang = df_lang[df_lang.lang == lang]

            if df_lang.empty:
                self._logger.error(
                    f"-- -- No documents found for language {lang}.")
                return 1

            corpus_txt_path = self._train_data_folder / f"corpus_{lang}.txt"

            self._logger.info(
                f"-- -- Creating Mallet {corpus_txt_path.as_posix()}...")

            with corpus_txt_path.open("w", encoding="utf8") as fout:
                for i, t in zip(df_lang.doc_id, df_lang.lemmas):
                    fout.write(f"{i} 0 {t}\n")

            self._logger.info(
                f"-- -- Mallet {corpus_txt_path.as_posix()} created.")

        if (self._train_data_folder / "corpus_lang1.txt").exists() and (self._train_data_folder / "corpus_lang2.txt").exists():
            return 2
        else:
            return 0

    def _prepare_mallet_input(
        self,
    ) -> int:
        """Assuming there is a 'corpus_es.txt' and 'corpus_en.txt' files in the model folder, prepare the input files for the LDA Mallet, i.e., generates the files 'corpus_lang1.mallet' and 'corpus_lang2.mallet'.

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

            cmd = self._mallet_path.as_posix() + \
                ' import-file --preserve-case --keep-sequence ' + \
                '--remove-stopwords --token-regex "' + self._token_regexp + \
                '" --print-output --input %s --output %s'
            cmd = cmd % (corpus_txt_path, corpus_mallet)

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
        df_path: pathlib.Path,
    ) -> int:
        """Assuming there is a 'corpus_es.mallet' and 'corpus_en.mallet' files in the model folder, trains two LDA Mallet Topic Models, one for each language in the corpus.

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
        for lang in [self._lang1, self._lang2]:

            # Get the training data
            mallet_input_file = \
                self._mallet_input_folder / f"corpus_{lang}.mallet"

            # Create folder for saving Mallet output for each language
            mallet_out_folder_lang = self._mallet_out_folder / f"{lang}"
            mallet_out_folder_lang.mkdir(exist_ok=True)

            # Create the Mallet configuration file
            config_mallet = mallet_out_folder_lang / f"config_{lang}.mallet"

            with config_mallet.open('w', encoding='utf8') as fout:
                fout.write(
                    'input = ' + mallet_input_file.resolve().as_posix() + '\n')
                fout.write('num-topics = ' + str(self._num_topics) + '\n')
                fout.write('alpha = ' + str(self._alpha) + '\n')
                fout.write('optimize-interval = ' +
                           str(self._optimize_interval) + '\n')
                fout.write('num-threads = ' + str(self._num_threads) + '\n')
                fout.write('num-iterations = ' +
                           str(self._num_iterations) + '\n')
                fout.write('doc-topics-threshold = ' +
                           str(self._doc_topic_thr) + '\n')
                fout.write('output-state = ' + mallet_out_folder_lang.joinpath(
                    'topic-state.gz').resolve().as_posix() + '\n')
                fout.write('output-doc-topics = ' +
                           mallet_out_folder_lang.joinpath('doc-topics.txt').resolve().as_posix() + '\n')
                fout.write('word-topic-counts-file = ' +
                           mallet_out_folder_lang.joinpath('word-topic-counts.txt').resolve().as_posix() + '\n')
                fout.write('diagnostics-file = ' +
                           mallet_out_folder_lang.joinpath('diagnostics.xml ').resolve().as_posix() + '\n')
                fout.write('xml-topic-report = ' +
                           mallet_out_folder_lang.joinpath('topic-report.xml').resolve().as_posix() + '\n')
                fout.write('output-topic-keys = ' +
                           mallet_out_folder_lang.joinpath('topickeys.txt').resolve().as_posix() + '\n')
                fout.write('inferencer-filename = ' +
                           mallet_out_folder_lang.joinpath('inferencer.mallet').resolve().as_posix() + '\n')

            cmd = self._mallet_path.as_posix() + \
                ' train-topics --config ' + str(config_mallet)

            try:
                self._logger.info(
                    f'-- -- Training LDA Mallet topic model. Command is {cmd}')
                check_output(args=cmd, shell=True)
            except:
                self._logger.error(
                    '-- -- Model training failed. Revise command')
                return
            
            self._get_more_info(mallet_out_folder_lang)

        return self._mallet_out_folder

    
    def _get_more_info(
        self,
        modelFolder: pathlib.Path
    ):

        # Load thetas matrix
        thetas_file = modelFolder.joinpath('doc-topics.txt')
        cols = [k for k in np.arange(2, self._num_topics + 2)]

        # Sparsification of thetas matrix
        self._logger.info('-- -- Sparsifying doc-topics matrix')
        thetas32 = np.loadtxt(thetas_file, delimiter='\t',
                              dtype=np.float32, usecols=cols)
        # Set to zeros all thetas below threshold, and renormalize
        thetas32[thetas32 < self._thetas_thr] = 0
        thetas32 = normalize(thetas32, axis=1, norm='l1')
        print(thetas32.shape)
        thetas = sparse.csr_matrix(thetas32, copy=True)

        # Recalculate topic weights to avoid errors due to sparsification
        alphas = np.asarray(np.mean(thetas32, axis=0)).ravel()

        # Create vocabulary files and calculate beta matrix
        # A vocabulary is available with words provided by the Count Vectorizer object, but the new files need the order used by mallet
        wtcFile = modelFolder.joinpath('word-topic-counts.txt')
        vocab_size = file_lines(wtcFile)
        betas = np.zeros((self._num_topics, vocab_size))
        vocab = []
        term_freq = np.zeros((vocab_size,))

        with wtcFile.open('r', encoding='utf8') as fin:
            for i, line in enumerate(fin):
                elements = line.split()
                vocab.append(elements[1])
                for counts in elements[2:]:
                    tpc = int(counts.split(':')[0])
                    cnt = int(counts.split(':')[1])
                    betas[tpc, i] += cnt
                    term_freq[i] += cnt
        betas = normalize(betas, axis=1, norm='l1')
        
        # Save vocabulary and frequencies
        vocabfreq_file = modelFolder.joinpath('vocab_freq.txt')
        with vocabfreq_file.open('w', encoding='utf8') as fout:
            [fout.write(el[0] + '\t' + str(int(el[1])) + '\n')
             for el in zip(vocab, term_freq)]
        self._logger.info('-- -- Mallet training: Vocabulary file generated')
        
        # Save thetas, alphas, and betas
        np.save(modelFolder.joinpath('alphas.npy'), alphas)
        np.save(modelFolder.joinpath('betas.npy'), betas)
        sparse.save_npz(modelFolder.joinpath('thetas.npz'), thetas)
        
        # Generate also pyLDAvisualization
        # pyLDAvis currently raises some Deprecation warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyLDAvis

        # We will compute the visualization using ndocs random documents
        # In case the model has gone through topic deletion, we may have rows
        # in the thetas matrix that sum up to zero (active topics have been
        # removed for these problematic documents). We need to take this into
        # account
        ndocs = 10000
        validDocs = np.sum(thetas.toarray(), axis=1) > 0
        nValidDocs = np.sum(validDocs)
        if ndocs > nValidDocs:
            ndocs = nValidDocs
        perm = np.sort(np.random.permutation(nValidDocs)[:ndocs])
        # We consider all documents are equally important
        doc_len = ndocs * [1]
        vocabfreq = np.round(ndocs*(alphas.dot(betas))).astype(int)
        vis_data = pyLDAvis.prepare(
            betas,
            thetas[validDocs, ][perm, ].toarray(),
            doc_len,
            vocab,
            vocabfreq,
            lambda_step=0.05,
            sort_topics=False,
            n_jobs=-1)
        with modelFolder.joinpath("pyLDAvis.html").open("w") as f:
            pyLDAvis.save_html(vis_data, f)