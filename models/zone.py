"""Zone enum + fixed headcount per zone type (single source for population ints)."""

from enum import Enum


class ZoneType(Enum):
    """Land-use labels on the grid."""

    RESIDENTIAL = "R"
    INDUSTRIAL  = "I"
    SCHOOL      = "S"
    POWER_PLANT = "W"
    HOSPITAL    = "H"
    DEPOT       = "D"

ZONE_POPULATION: dict[ZoneType, int] = {
    ZoneType.RESIDENTIAL: 1_000,
    ZoneType.SCHOOL:         800,
    ZoneType.HOSPITAL:       600,
    ZoneType.INDUSTRIAL:     300,
    ZoneType.POWER_PLANT:     75,
    ZoneType.DEPOT:            0,
}
