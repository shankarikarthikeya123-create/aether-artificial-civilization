from datetime import datetime, timedelta, timezone

from .models import WorldState, Location, Resource


# India Standard Time is UTC+05:30 year-round; using a fixed offset avoids external tzdata dependency.
AETHER_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


class CivilizationEngine:

    def __init__(self):
        self.world = self._create_initial_world()
        self.sync_real_time()

    def _create_initial_world(self) -> WorldState:

        locations = [
            Location(
                id="capital",
                name="Aether Central",
                x=50,
                y=50,
                location_type="city",
            ),
            Location(
                id="north",
                name="North District",
                x=50,
                y=20,
                location_type="district",
            ),
            Location(
                id="industrial",
                name="Industrial District",
                x=75,
                y=55,
                location_type="industry",
            ),
            Location(
                id="farmland",
                name="Green Fields",
                x=25,
                y=70,
                location_type="farmland",
            ),
            Location(
                id="harbor",
                name="Aether Harbor",
                x=75,
                y=80,
                location_type="harbor",
            ),
            Location(
                id="forest",
                name="Whispering Forest",
                x=20,
                y=25,
                location_type="forest",
            ),
            Location(
                id="mountains",
                name="Northern Mountains",
                x=80,
                y=15,
                location_type="mountains",
            ),
        ]

        resources = [
            Resource(
                name="food",
                amount=1000,
                capacity=2000,
            ),
            Resource(
                name="water",
                amount=2000,
                capacity=4000,
            ),
            Resource(
                name="energy",
                amount=1500,
                capacity=3000,
            ),
            Resource(
                name="money",
                amount=100000,
                capacity=1000000,
            ),
            Resource(
                name="knowledge",
                amount=100,
                capacity=10000,
            ),
        ]

        return WorldState(
            name="AETHER",
            date="",
            time="",
            day_of_week="",
            timezone="Asia/Kolkata",
            day=1,
            hour=0,
            minute=0,
            locations=locations,
            resources=resources,
            population=100,
            events=[
                "AETHER civilization initialized.",
            ],
        )

    def sync_real_time(self):

        now = datetime.now(AETHER_TIMEZONE)

        self.world.date = now.strftime("%Y-%m-%d")
        self.world.time = now.strftime("%H:%M:%S")
        self.world.day_of_week = now.strftime("%A")

        # Compatibility values used by the existing cognition system.
        self.world.day = now.day
        self.world.hour = now.hour
        self.world.minute = now.minute

        self.world.timezone = "Asia/Kolkata"

        return self.world

    def get_world(self) -> WorldState:

        # Always keep the displayed AETHER clock synchronized
        # with real-world time.
        self.sync_real_time()

        return self.world

    def advance_time(self, minutes: int = 0):

        # AETHER no longer advances an artificial clock.
        #
        # The civilization's official clock follows real time.
        #
        # The parameter remains for compatibility with the existing
        # simulation API.

        self.sync_real_time()

        return self.world