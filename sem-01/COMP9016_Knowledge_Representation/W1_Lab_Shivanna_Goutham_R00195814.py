from statistics import mean
from ipythonblocks import BlockGrid
from IPython.display import HTML, display, clear_output
from time import sleep

import random
import copy
import collections
import numbers


# ______________________________________________________________________________
# Util code
# ______________________________________________________________________________

orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def distance_squared(a, b):
    """The square of the distance between two (x, y) points."""
    xA, yA = a
    xB, yB = b
    return (xA - xB) ** 2 + (yA - yB) ** 2


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]

# ______________________________________________________________________________
# Code related to AGENT
# ______________________________________________________________________________


class Thing:
    """This represents any physical object that can appear in an Environment.
    You subclass Thing to get the things you want. Each thing can have a
    .__name__  slot (used for output only)."""

    def __repr__(self):
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))

    def is_alive(self):
        """Things that are 'alive' should return true."""
        return hasattr(self, 'alive') and self.alive

    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state.")


class Agent(Thing):
    """An Agent is a subclass of Thing with one required instance attribute
    (aka slot), .program, which should hold a function that takes one argument,
    the percept, and returns an action. (What counts as a percept or action
    will depend on the specific environment in which the agent exists.)
    Note that 'program' is a slot, not a method. If it were a method, then the
    program could 'cheat' and look at aspects of the agent. It's not supposed
    to do that: the program can only look at the percepts. An agent program
    that needs a model of the world (and of the agent itself) will have to
    build and maintain its own model. There is an optional slot, .performance,
    which is a number giving the performance measure of the agent in its
    environment."""

    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0
        if program is None or not isinstance(program, collections.abc.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(
                self.__class__.__name__))

            def program(percept):
                return eval(input('Percept={}; action? '.format(percept)))

        self.program = program

    def can_grab(self, thing):
        """Return True if this agent can grab this thing.
        Override for appropriate subclasses of Agent and Thing."""
        return False


def TraceAgent(agent):
    """Wrap the agent's program to print its input and output. This will let
    you see what the agent is doing in the environment."""
    old_program = agent.program

    def new_program(percept):
        action = old_program(percept)
        print('{} perceives {} and does {}'.format(agent, percept, action))
        return action

    agent.program = new_program
    return agent


class Dirt(Thing):
    pass

# ______________________________________________________________________________
# Code related to ENVIRONMENT
# ______________________________________________________________________________


class Obstacle(Thing):
    """Something that can cause a bump, preventing an agent from
    moving into the same square it's in."""
    pass


class Wall(Obstacle):
    pass


class Environment:
    """Abstract class representing an Environment. 'Real' Environment classes
    inherit from this. Your Environment will typically need to implement:
        percept:           Define the percept that an agent sees.
        execute_action:    Define the effects of executing an action.
                           Also update the agent.performance slot.
    The environment keeps a list of .things and .agents (which is a subset
    of .things). Each agent has a .performance slot, initialized to 0.
    Each thing has a .location slot, even though some environments may not
    need this."""

    def __init__(self):
        self.things = []
        self.agents = []

    def thing_classes(self):
        return []  # List of classes that can go into environment

    def percept(self, agent):
        """Return the percept that the agent sees at this point. (Implement this.)"""
        raise NotImplementedError

    def execute_action(self, agent, action):
        """Change the world to reflect this action. (Implement this.)"""
        raise NotImplementedError

    def default_location(self, thing):
        """Default location to place a new thing with unspecified location."""
        return None

    def exogenous_change(self):
        """If there is spontaneous change in the world, override this."""
        pass

    def is_done(self):
        """By default, we're done when we can't find a live agent."""
        return not any(agent.is_alive() for agent in self.agents)

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            for agent in self.agents:
                if agent.alive:
                    actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            for (agent, action) in zip(self.agents, actions):
                self.execute_action(agent, action)
            self.exogenous_change()

    def run(self, steps=1000):
        """Run the Environment for given number of time steps."""
        for step in range(steps):
            print(f' Run: {step+1} / {steps} '.center(80, '-'))
            if self.is_done():
                return
            self.step()

    def list_things_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        if isinstance(location, numbers.Number):
            return [thing for thing in self.things
                    if thing.location == location and isinstance(thing, tclass)]
        return [thing for thing in self.things
                if all(x == y for x, y in zip(thing.location, location)) and isinstance(thing, tclass)]

    def some_things_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_things_at(location, tclass) != []

    def add_thing(self, thing, location=None):
        """Add a thing to the environment, setting its location. For
        convenience, if thing is an agent program we make a new agent
        for it. (Shouldn't need to override this.)"""
        if not isinstance(thing, Thing):
            thing = Agent(thing)
        if thing in self.things:
            print("Can't add the same thing twice")
        else:
            thing.location = location if location is not None else self.default_location(
                thing)
            self.things.append(thing)
            if isinstance(thing, Agent):
                thing.performance = 0
                self.agents.append(thing)

    def delete_thing(self, thing):
        """Remove a thing from the environment."""
        try:
            self.things.remove(thing)
        except ValueError as e:
            print(e)
            print("  in Environment delete_thing")
            print("  Thing to be removed: {} at {}".format(thing, thing.location))
            print("  from list: {}".format(
                [(thing, thing.location) for thing in self.things]))
        if thing in self.agents:
            self.agents.remove(thing)


class XYEnvironment(Environment):
    """This class is for environments on a 2D plane, with locations
    labelled by (x, y) points, either discrete or continuous.

    Agents perceive things within a radius. Each agent in the
    environment has a .location slot which should be a location such
    as (0, 1), and a .holding slot, which should be a list of things
    that are held."""

    def __init__(self, width=10, height=10):
        super().__init__()

        self.width = width
        self.height = height
        self.observers = []
        # Sets iteration start and end (no walls).
        self.x_start, self.y_start = (0, 0)
        self.x_end, self.y_end = (self.width, self.height)

    perceptible_distance = 1

    def things_near(self, location, radius=None):
        """Return all things within radius of location."""
        if radius is None:
            radius = self.perceptible_distance
        radius2 = radius * radius
        return [(thing, radius2 - distance_squared(location, thing.location))
                for thing in self.things if distance_squared(
                location, thing.location) <= radius2]

    def percept(self, agent):
        """By default, agent perceives things within a default radius."""
        return self.things_near(agent.location)

    def execute_action(self, agent, action):
        agent.bump = False
        if action == 'TurnRight':
            agent.direction += Direction.R
        elif action == 'TurnLeft':
            agent.direction += Direction.L
        elif action == 'Forward':
            agent.bump = self.move_to(
                agent, agent.direction.move_forward(agent.location))
        elif action == 'Grab':
            things = [thing for thing in self.list_things_at(
                agent.location) if agent.can_grab(thing)]
            if things:
                agent.holding.append(things[0])
                print("Grabbing ", things[0].__class__.__name__)
                self.delete_thing(things[0])
        elif action == 'Release':
            if agent.holding:
                dropped = agent.holding.pop()
                print("Dropping ", dropped.__class__.__name__)
                self.add_thing(dropped, location=agent.location)

    def default_location(self, thing):
        location = self.random_location_inbounds()
        while self.some_things_at(location, Obstacle):
            # we will find a random location with no obstacles
            location = self.random_location_inbounds()
        return location

    def move_to(self, thing, destination):
        """Move a thing to a new location. Returns True on success or False if there is an Obstacle.
        If thing is holding anything, they move with him."""
        thing.bump = self.some_things_at(destination, Obstacle)
        if not thing.bump:
            thing.location = destination
            for o in self.observers:
                o.thing_moved(thing)
            for t in thing.holding:
                self.delete_thing(t)
                self.add_thing(t, destination)
                t.location = destination
        return thing.bump

    def add_thing(self, thing, location=None, exclude_duplicate_class_items=False):
        """Add things to the world. If (exclude_duplicate_class_items) then the item won't be
        added if the location has at least one item of the same class."""
        if location is None:
            super().add_thing(thing)
        elif self.is_inbounds(location):
            if (exclude_duplicate_class_items and
                    any(isinstance(t, thing.__class__) for t in self.list_things_at(location))):
                return
            super().add_thing(thing, location)

    def is_inbounds(self, location):
        """Checks to make sure that the location is inbounds (within walls if we have walls)"""
        x, y = location
        return not (x < self.x_start or x > self.x_end or y < self.y_start or y > self.y_end)

    def random_location_inbounds(self, exclude=None):
        """Returns a random location that is inbounds (within walls if we have walls)"""
        location = (random.randint(self.x_start, self.x_end),
                    random.randint(self.y_start, self.y_end))
        if exclude is not None:
            while location == exclude:
                location = (random.randint(self.x_start, self.x_end),
                            random.randint(self.y_start, self.y_end))
        return location

    def delete_thing(self, thing):
        """Deletes thing, and everything it is holding (if thing is an agent)"""
        if isinstance(thing, Agent):
            del thing.holding

        super().delete_thing(thing)
        for obs in self.observers:
            obs.thing_deleted(thing)

    def add_walls(self):
        """Put walls around the entire perimeter of the grid."""
        for x in range(self.width):
            self.add_thing(Wall(), (x, 0))
            self.add_thing(Wall(), (x, self.height - 1))
        for y in range(1, self.height - 1):
            self.add_thing(Wall(), (0, y))
            self.add_thing(Wall(), (self.width - 1, y))

        # Updates iteration start and end (with walls).
        self.x_start, self.y_start = (1, 1)
        self.x_end, self.y_end = (self.width - 1, self.height - 1)

    def add_observer(self, observer):
        """Adds an observer to the list of observers.
        An observer is typically an EnvGUI.

        Each observer is notified of changes in move_to and add_thing,
        by calling the observer's methods thing_moved(thing)
        and thing_added(thing, loc)."""
        self.observers.append(observer)

    def turn_heading(self, heading, inc):
        """Return the heading to the left (inc=+1) or right (inc=-1) of heading."""
        return turn_heading(heading, inc)


class Direction:
    """A direction class for agents that want to move in a 2D plane
        Usage:
            d = Direction("down")
            To change directions:
            d = d + "right" or d = d + Direction.R #Both do the same thing
            Note that the argument to __add__ must be a string and not a Direction object.
            Also, it (the argument) can only be right or left."""

    R = "right"
    L = "left"
    U = "up"
    D = "down"

    def __init__(self, direction):
        self.direction = direction

    def __add__(self, heading):
        """
        >>> d = Direction('right')
        >>> l1 = d.__add__(Direction.L)
        >>> l2 = d.__add__(Direction.R)
        >>> l1.direction
        'up'
        >>> l2.direction
        'down'
        >>> d = Direction('down')
        >>> l1 = d.__add__('right')
        >>> l2 = d.__add__('left')
        >>> l1.direction == Direction.L
        True
        >>> l2.direction == Direction.R
        True
        """
        if self.direction == self.R:
            return {
                self.R: Direction(self.D),
                self.L: Direction(self.U),
            }.get(heading, None)
        elif self.direction == self.L:
            return {
                self.R: Direction(self.U),
                self.L: Direction(self.D),
            }.get(heading, None)
        elif self.direction == self.U:
            return {
                self.R: Direction(self.R),
                self.L: Direction(self.L),
            }.get(heading, None)
        elif self.direction == self.D:
            return {
                self.R: Direction(self.L),
                self.L: Direction(self.R),
            }.get(heading, None)

    def move_forward(self, from_location):
        """
        >>> d = Direction('up')
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (0, -1)
        >>> d = Direction(Direction.R)
        >>> l1 = d.move_forward((0, 0))
        >>> l1
        (1, 0)
        """
        # get the iterable class to return
        iclass = from_location.__class__
        x, y = from_location
        if self.direction == self.R:
            return iclass((x + 1, y))
        elif self.direction == self.L:
            return iclass((x - 1, y))
        elif self.direction == self.U:
            return iclass((x, y - 1))
        elif self.direction == self.D:
            return iclass((x, y + 1))


class TrivialVacuumEnvironment(Environment):
    """This environment has two locations, A and B. Each can be Dirty
    or Clean. The agent perceives its location and the location's
    status. This serves as an example of how to implement a simple
    Environment."""

    def __init__(self):
        super().__init__()
        self.status = {loc_A: random.choice(['Clean', 'Dirty']),
                       loc_B: random.choice(['Clean', 'Dirty'])}

    def thing_classes(self):
        return [Wall, Dirt, TableDrivenVacuumAgent]

    def percept(self, agent):
        """Returns the agent's location, and the location status (Dirty/Clean)."""
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        if additional_log == 1:
            print(f'Agent location: {agent.location}')
            print('Agent status and performance score before performing the action')
            print(f'Env.status: {self.status}')
            print(f'Agent Performance: {agent.performance}')
        """Change agent's location and/or location's status; track performance.
        Score 10 for each dirt cleaned; -1 for each move."""
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Suck':
            if self.status[agent.location] == 'Dirty':
                agent.performance += 10
            self.status[agent.location] = 'Clean'

        print(f'\nPerforming Action: {action}\n')
        if additional_log == 1:
            print('Agent status and performance score after performing the action')
        print(f'Agent location: {agent.location}')
        print(f'Env.status: {self.status}')
        print(f'Agent Performance: {agent.performance}')

    def default_location(self, thing):
        """Agents start in either location at random."""
        return random.choice([loc_A, loc_B])

# ______________________________________________________________________________
# Code related to TableDrivenAgent
# ______________________________________________________________________________


def TableDrivenAgentProgram(table):
    """
    [Figure 2.7]
    This agent selects an action based on the percept sequence.
    It is practical only for tiny domains.
    To customize it, provide as table a dictionary of all
    {percept_sequence:action} pairs.
    """
    percepts = []

    def program(percept):
        percepts.append(percept)
        action = table.get(tuple(percepts))
        return action

    return program


loc_A, loc_B = (0, 0), (1, 0)  # The two locations for the Vacuum world


def TableDrivenVacuumAgent():
    """Tabular approach towards vacuum world as mentioned in [Figure 2.3]
    >>> agent = TableDrivenVacuumAgent()
    >>> environment = TrivialVacuumEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    >>> environment.status == {(1,0):'Clean' , (0,0) : 'Clean'}
    True
    """
    table = {((loc_A, 'Clean'),): 'Right',
             ((loc_A, 'Dirty'),): 'Suck',
             ((loc_B, 'Clean'),): 'Left',
             ((loc_B, 'Dirty'),): 'Suck',
             ((loc_A, 'Dirty'), (loc_A, 'Clean')): 'Right',
             ((loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean')): 'Left',
             ((loc_A, 'Dirty'), (loc_A, 'Clean'), (loc_B, 'Dirty')): 'Suck',
             ((loc_B, 'Dirty'), (loc_B, 'Clean'), (loc_A, 'Dirty')): 'Suck'}
    return Agent(TableDrivenAgentProgram(table))


# ==============================================================================
# ----------------------------- Farmers Dilemma --------------------------------
# ==============================================================================

def TableDrivenFarmerAgent():
    """Tabular approach towards farmer dilemma world
    >>> agent = TableDrivenFarmerAgent()
    >>> environment = TrivialFarmerEnvironment()
    >>> environment.add_thing(agent)
    >>> environment.run()
    """

    table = {((loc_A, 'ChickenFeedFox'),): 'Chicken Right',
             ((loc_A, 'ChickenFeedFox'), (loc_B, 'Chicken'),): 'Left',
             ((loc_A, 'ChickenFeedFox'), (loc_B, 'Chicken'), (loc_A, 'FeedFox')): 'Fox Right',
             ((loc_A, 'ChickenFeedFox'), (loc_B, 'Chicken'), (loc_A, 'FeedFox'), (loc_B, 'ChickenFox')): 'Chicken Left',
             ((loc_A, 'ChickenFeedFox'), (loc_B, 'Chicken'), (loc_A, 'FeedFox'), (loc_B, 'ChickenFox'), (loc_A, 'FeedChicken')): 'Feed Right',
             ((loc_A, 'ChickenFeedFox'), (loc_B, 'Chicken'), (loc_A, 'FeedFox'), (loc_B, 'ChickenFox'), (loc_A, 'FeedChicken'), (loc_B, 'FoxFeed')): 'Left',
             ((loc_A, 'ChickenFeedFox'), (loc_B, 'Chicken'), (loc_A, 'FeedFox'), (loc_B, 'ChickenFox'), (loc_A, 'FeedChicken'), (loc_B, 'FoxFeed'), (loc_A, 'Chicken')): 'Chicken Right', }
    return Agent(TableDrivenAgentProgram(table))


class TrivialFarmerEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self.status = {loc_A: 'ChickenFeedFox',
                       loc_B: ''}

    def thing_classes(self):
        return [Wall, Dirt, TableDrivenFarmerAgent]

    def percept(self, agent):
        return agent.location, self.status[agent.location]

    def execute_action(self, agent, action):
        if additional_log == 1:
            print(f'Agent location: {agent.location}')
            print('Agent status and performance score before performing the action')
            print(f'Env.status: {self.status}')
            print(f'Agent Performance: {agent.performance}')
        if action == 'Right':
            agent.location = loc_B
            agent.performance -= 1
        elif action == 'Left':
            agent.location = loc_A
            agent.performance -= 1
        elif action == 'Chicken Right':
            agent.location = loc_B
            self.status[loc_A] = self.status[loc_A].replace('Chicken', '')
            self.status[loc_B] += 'Chicken'
            agent.performance += 10
        elif action == 'Chicken Left':
            agent.location = loc_A
            self.status[loc_A] += 'Chicken'
            self.status[loc_B] = self.status[loc_B].replace('Chicken', '')
            agent.performance -= 5
        elif action == 'Fox Right':
            agent.location = loc_B
            self.status[loc_A] = self.status[loc_A].replace('Fox', '')
            self.status[loc_B] += 'Fox'
            agent.performance += 10
        elif action == 'Feed Right':
            agent.location = loc_B
            self.status[loc_A] = self.status[loc_A].replace('Feed', '')
            self.status[loc_B] += 'Feed'
            agent.performance += 10
        elif action == 'NoOp':
            pass

        print(f'\nPerforming Action: {action}\n')
        if additional_log == 1:
            print('Agent status and performance score after performing the action')
        print(f'Agent location: {agent.location}')
        print(f'Env.status: {self.status}')
        print(f'Agent Performance: {agent.performance}')

    def default_location(self, thing):
        return loc_A


# ------------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------------

additional_log = 1


def runEnvironment(agent=Agent, environment=Environment, runs=int):
    environment.add_thing(agent)
    environment.run(steps=runs)


if __name__ == '__main__':
    print('\n'*20)
    # Main execution - Table Driven Vacuum Agent
    print('='*80)
    print(' Vacuum Agent '.center(80, '='))
    print('='*80)
    for run in [2, 4, 8]:
        vacuum_env = TrivialVacuumEnvironment()
        tdva = TableDrivenVacuumAgent()
        runEnvironment(tdva, vacuum_env, int(run))
        print('\n')
        print('='*80)
    """
    Q.  What is the optimal status and sequence of actions in the context of the agents performance?
    A.  The optimal (initial) status would be {(0, 0): 'Dirty', (1, 0): 'Dirty'}, and the run would be set to 3.
        For example:  With the above status, the agent would clean the location where it has initially spawned and gains a performance of 10.
                      Then the agent will navigate to the other location, which reduces the performance by 1 and performance now becomes 9.
                      Lastly, the agent cleans the new location to which it moved and gains another 10 points where the total performance would be 19.

    Q.  What is the least optimal status and sequence of actions in the context of the agents performance?
    A.  So, the least optimal would be the opposite of the above scenario, meaning, when the initial status is {(0, 0): 'Clean', (1, 0): 'Clean'}
        Here, the agent would navigate to the other location without performing any action on it location where it has spawned, which results in the performance becoming -1.
        And, now as the new location it has moved to is also clean, there is no action to perform and the performance will remain -1 (if the run is set to 2)
    """

    # Main execution - Table Driven Vacuum Agent
    print(' Farmer Dilemma '.center(80, '='))
    print('='*80)
    farmer_env = TrivialFarmerEnvironment()
    tdfa = TableDrivenFarmerAgent()
    runEnvironment(tdfa, farmer_env, int(10))
    print('='*80)
    print('\n'*5)
