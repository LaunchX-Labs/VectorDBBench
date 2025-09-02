# import os
# from typing import Annotated, Unpack

# import click
# from pydantic import SecretStr

# from vectordb_bench.backend.clients import DB
# from vectordb_bench.backend.clients.api import MetricType

# from ....cli.cli import (
#     CommonTypedDict,
#     IVFFlatTypedDict,
#     cli,
#     click_parameter_decorators_from_typed_dict,
#     get_custom_case_config,
#     run,
# )


# class FacebookMyRocksTypedDict(CommonTypedDict):
#     user_name: Annotated[
#         str,
#         click.option("--user-name", type=str, help="MySQL username", default="root", show_default=True),
#     ]
#     password: Annotated[
#         str,
#         click.option(
#             "--password",
#             type=str,
#             help="MySQL database password",
#             default=lambda: os.environ.get("MYSQL_PASSWORD", ""),
#             show_default="$MYSQL_PASSWORD",
#         ),
#     ]

#     host: Annotated[str, click.option("--host", type=str, help="MySQL host", default="localhost", show_default=True)]
#     port: Annotated[
#         int,
#         click.option(
#             "--port",
#             type=int,
#             help="MySQL database port",
#             default=3306,
#             show_default=True,
#             required=False,
#         ),
#     ]
#     db_name: Annotated[str, click.option("--db-name", type=str, help="Database name", required=True)]
    
#     vector_type: Annotated[
#         str,
#         click.option(
#             "--vector-type",
#             type=click.Choice(["BLOB", "JSON"]),
#             help="Vector storage type (BLOB recommended for efficiency)",
#             default="BLOB",
#             show_default=True,
#         ),
#     ]
    
#     trained_index_table: Annotated[
#         str | None,
#         click.option(
#             "--trained-index-table",
#             type=str,
#             help="Table name for storing trained IVF index data (required for IVF indexes)",
#             required=False,
#         ),
#     ]
    
#     trained_index_id: Annotated[
#         int,
#         click.option(
#             "--trained-index-id",
#             type=int,
#             help="ID for the trained index (for IVF indexes)",
#             default=1,
#             show_default=True,
#         ),
#     ]


# class FacebookMyRocksFlatTypedDict(FacebookMyRocksTypedDict):
#     pass


# class FacebookMyRocksIVFFlatTypedDict(FacebookMyRocksTypedDict):
#     nlist: Annotated[
#         int | None,
#         click.option(
#             "--nlist",
#             type=int,
#             help="Number of clusters/centroids for IVF index",
#             required=False,
#         ),
#     ]
#     nprobe: Annotated[
#         int | None,
#         click.option(
#             "--nprobe", 
#             type=int,
#             help="Number of clusters to search during query",
#             required=False,
#         ),
#     ]


# class FacebookMyRocksIVFPQTypedDict(FacebookMyRocksIVFFlatTypedDict):
#     m: Annotated[
#         int | None,
#         click.option(
#             "--m",
#             type=int,
#             help="Number of subquantizers for Product Quantization",
#             required=False,
#         ),
#     ]
#     nbits: Annotated[
#         int | None,
#         click.option(
#             "--nbits",
#             type=int,
#             help="Number of bits per subquantizer",
#             default=8,
#             show_default=True,
#         ),
#     ]


# @cli.command()
# @click_parameter_decorators_from_typed_dict(FacebookMyRocksFlatTypedDict)
# def FacebookMyRocksFlat(
#     **parameters: Unpack[FacebookMyRocksFlatTypedDict],
# ):
#     from .config import FacebookMyRocksConfig, FacebookMyRocksFlatConfig

#     parameters["custom_case"] = get_custom_case_config(parameters)
#     run(
#         db=DB.FacebookMyRocks,
#         db_config=FacebookMyRocksConfig(
#             db_label=parameters["db_label"],
#             user_name=SecretStr(parameters["user_name"]),
#             password=SecretStr(parameters["password"]),
#             host=parameters["host"],
#             port=parameters["port"],
#             db_name=parameters["db_name"],
#         ),
#         db_case_config=FacebookMyRocksFlatConfig(
#             metric_type=None,
#             vector_type=parameters["vector_type"],
#         ),
#         **parameters,
#     )


# @cli.command()
# @click_parameter_decorators_from_typed_dict(FacebookMyRocksIVFFlatTypedDict)
# def FacebookMyRocksIVFFlat(
#     **parameters: Unpack[FacebookMyRocksIVFFlatTypedDict],
# ):
#     from .config import FacebookMyRocksConfig, FacebookMyRocksIVFFlatConfig

#     parameters["custom_case"] = get_custom_case_config(parameters)
#     run(
#         db=DB.FacebookMyRocks,
#         db_config=FacebookMyRocksConfig(
#             db_label=parameters["db_label"],
#             user_name=SecretStr(parameters["user_name"]),
#             password=SecretStr(parameters["password"]),
#             host=parameters["host"],
#             port=parameters["port"],
#             db_name=parameters["db_name"],
#         ),
#         db_case_config=FacebookMyRocksIVFFlatConfig(
#             metric_type=None,
#             vector_type=parameters["vector_type"],
#             nlist=parameters["nlist"],
#             nprobe=parameters["nprobe"],
#             trained_index_table=parameters["trained_index_table"],
#             trained_index_id=parameters["trained_index_id"],
#         ),
#         **parameters,
#     )


# @cli.command()
# @click_parameter_decorators_from_typed_dict(FacebookMyRocksIVFPQTypedDict)
# def FacebookMyRocksIVFPQ(
#     **parameters: Unpack[FacebookMyRocksIVFPQTypedDict],
# ):
#     from .config import FacebookMyRocksConfig, FacebookMyRocksIVFPQConfig

#     parameters["custom_case"] = get_custom_case_config(parameters)
#     run(
#         db=DB.FacebookMyRocks,
#         db_config=FacebookMyRocksConfig(
#             db_label=parameters["db_label"],
#             user_name=SecretStr(parameters["user_name"]),
#             password=SecretStr(parameters["password"]),
#             host=parameters["host"],
#             port=parameters["port"],
#             db_name=parameters["db_name"],
#         ),
#         db_case_config=FacebookMyRocksIVFPQConfig(
#             metric_type=None,
#             vector_type=parameters["vector_type"],
#             nlist=parameters["nlist"],
#             nprobe=parameters["nprobe"],
#             m=parameters["m"],
#             nbits=parameters["nbits"],
#             trained_index_table=parameters["trained_index_table"],
#             trained_index_id=parameters["trained_index_id"],
#         ),
#         **parameters,
#     )

import os
from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType

from ....cli.cli import (
    CommonTypedDict,
    IVFFlatTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class FacebookMyRocksTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="MySQL username", default="root", show_default=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="MySQL database password",
            default=lambda: os.environ.get("MYSQL_PASSWORD", ""),
            show_default="$MYSQL_PASSWORD",
        ),
    ]

    host: Annotated[str, click.option("--host", type=str, help="MySQL host", default="localhost", show_default=True)]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="MySQL database port",
            default=3306,
            show_default=True,
            required=False,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Database name", required=True)]
    
    vector_type: Annotated[
        str,
        click.option(
            "--vector-type",
            type=click.Choice(["BLOB", "JSON"]),
            help="Vector storage type (BLOB recommended for efficiency)",
            default="BLOB",
            show_default=True,
        ),
    ]
    
    trained_index_table: Annotated[
        str | None,
        click.option(
            "--trained-index-table",
            type=str,
            help="Table name for storing trained IVF index data (required for IVF indexes)",
            required=False,
        ),
    ]
    
    trained_index_id: Annotated[
        int,
        click.option(
            "--trained-index-id",
            type=int,
            help="ID for the trained index (for IVF indexes)",
            default=1,
            show_default=True,
        ),
    ]


class FacebookMyRocksFlatTypedDict(FacebookMyRocksTypedDict):
    pass


class FacebookMyRocksIVFFlatTypedDict(FacebookMyRocksTypedDict):
    nlist: Annotated[
        int | None,
        click.option(
            "--nlist",
            type=int,
            help="Number of clusters/centroids for IVF index",
            required=False,
        ),
    ]
    nprobe: Annotated[
        int | None,
        click.option(
            "--nprobe", 
            type=int,
            help="Number of clusters to search during query",
            required=False,
        ),
    ]


class FacebookMyRocksIVFPQTypedDict(FacebookMyRocksIVFFlatTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="Number of subquantizers for Product Quantization",
            required=False,
        ),
    ]
    nbits: Annotated[
        int | None,
        click.option(
            "--nbits",
            type=int,
            help="Number of bits per subquantizer",
            default=8,
            show_default=True,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(FacebookMyRocksFlatTypedDict)
def FacebookMyRocksFlat(
    **parameters: Unpack[FacebookMyRocksFlatTypedDict],
):
    from .config import FacebookMyRocksConfig, FacebookMyRocksFlatConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.FacebookMyRocks,
        db_config=FacebookMyRocksConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=FacebookMyRocksFlatConfig(
            metric_type=None,
            vector_type=parameters["vector_type"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(FacebookMyRocksIVFFlatTypedDict)
def FacebookMyRocksIVFFlat(
    **parameters: Unpack[FacebookMyRocksIVFFlatTypedDict],
):
    from .config import FacebookMyRocksConfig, FacebookMyRocksIVFFlatConfig

    # Validate that trained_index_table is provided for IVF indexes
    if not parameters.get("trained_index_table"):
        raise click.ClickException("--trained-index-table is required for IVF indexes")

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.FacebookMyRocks,
        db_config=FacebookMyRocksConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=FacebookMyRocksIVFFlatConfig(
            metric_type=None,
            vector_type=parameters["vector_type"],
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            trained_index_table=parameters["trained_index_table"],
            trained_index_id=parameters["trained_index_id"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(FacebookMyRocksIVFPQTypedDict)
def FacebookMyRocksIVFPQ(
    **parameters: Unpack[FacebookMyRocksIVFPQTypedDict],
):
    from .config import FacebookMyRocksConfig, FacebookMyRocksIVFPQConfig

    # Validate that trained_index_table is provided for IVF indexes
    if not parameters.get("trained_index_table"):
        raise click.ClickException("--trained-index-table is required for IVF indexes")

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.FacebookMyRocks,
        db_config=FacebookMyRocksConfig(
            db_label=parameters["db_label"],
            user_name=SecretStr(parameters["user_name"]),
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=FacebookMyRocksIVFPQConfig(
            metric_type=None,
            vector_type=parameters["vector_type"],
            nlist=parameters["nlist"],
            nprobe=parameters["nprobe"],
            m=parameters["m"],
            nbits=parameters["nbits"],
            trained_index_table=parameters["trained_index_table"],
            trained_index_id=parameters["trained_index_id"],
        ),
        **parameters,
    )