import logging
from pathlib import Path

from llama_index import Document
from llama_index.readers import JSONReader, StringIterableReader
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS


from llama_index.readers.base import BaseReader
from llama_index import Document

# import cv2
# import fastdeploy.vision as vision

from private_gpt.components.ingest.ocr import *
logger = logging.getLogger(__name__)

# model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
#                             "ppyoloe_crn_l_300e_coco/model.pdiparams",
#                             "ppyoloe_crn_l_300e_coco/infer_cfg.yml")

from rapidocr_paddle import RapidOCR

engine = RapidOCR()
class MyImageReader(BaseReader):
    def load_data(self, file, extra_info=None):
        result, elapse_list = engine(file)
        text = str(result)
        return [Document(text=text + "Foobar", extra_info=extra_info or {})]
    
# class MyPdfReader(BaseReader):
#     def load_data(self, file, extra_info=None):
#         result, elapse_list = engine(file)
#         text = str(result)
#         return [Document(text=text + "Foobar", extra_info=extra_info or {})]





# Patching the default file reader to support other file types
FILE_READER_CLS = DEFAULT_FILE_READER_CLS.copy()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
        ".jpg": MyImageReader,
        ".png": MyImageReader,
        ".jpeg": MyImageReader,
    }
)


logger.debug(f"----------> {FILE_READER_CLS}")




class IngestionHelper:
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

    @staticmethod
    def transform_file_into_documents(
        file_name: str, file_data: Path
    ) -> list[Document]:
        documents = IngestionHelper._load_file_to_documents(file_name, file_data)
        for document in documents:
            document.metadata["file_name"] = file_name
        IngestionHelper._exclude_metadata(documents)
        return documents

    @staticmethod
    def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
        logger.debug("Transforming file_name=%s into documents", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            return string_reader.load_data([file_data.read_text()])

        logger.debug("Specific reader found for extension=%s", extension)
        return reader_cls().load_data(file_data)

    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> None:
        logger.debug("Excluding metadata from count=%s documents", len(documents))
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
