from datetime import date, datetime, time
from typing import Annotated, Any, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel
from pydantic.fields import Field

from pydantic_rng import configure_rng, generate

# ----------------------------
# TEST MODELS
# ----------------------------


class AllTypesModel(BaseModel):
    b: bool
    i: int
    f: float
    s: str
    bts: bytes
    d: date
    t: time
    dt: datetime
    o: Optional[int]
    l_int: List[int]
    s_str: Set[str]
    tup: Tuple[int, str, bytes]
    l_any: List[Any]
    lit: Literal["a", "b", "c"]


class CompoundModel(BaseModel):
    a1: list[str]
    a2: dict[str, int]
    nested: list[dict[str, float]]
    maybe: Optional[int]


class ConstrainedModel(BaseModel):
    f: Annotated[float, Field(ge=0, le=100)]
    s: Annotated[str, Field(max_length=100)]


def test_all_types():
    generate(AllTypesModel)


def test_compound_type():
    generate(CompoundModel)


def test_constrained_model():
    generate(ConstrainedModel)


def test_configure_rng_effect():
    configure_rng(numeric_min=10, numeric_max=20, min_str_length=5, max_str_length=8)

    class Model(BaseModel):
        i: int
        s: str
        b: bytes

    inst = generate(Model)
    assert 10 <= inst.i <= 20
    assert 5 <= len(inst.s) <= 8
    assert 4 <= len(inst.b) <= 100
