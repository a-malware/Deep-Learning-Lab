from scipy.spatial import distance
from collections import deque

from .enums import Event, HandPosition, targets
from .hand import Hand


class Deque:
    def __init__(self, maxlen=30, min_frames=20):
        self.maxlen = maxlen
        self._deque = []
        self.action = None
        self.min_absolute_distance = 1.5
        self.min_frames = min_frames
        self.action_deque = deque(maxlen=5)

    def __len__(self):
        return len(self._deque)

    def index_position(self, x):
        for i in range(len(self._deque)):
            if self._deque[i].position == x:
                return i

    def index_gesture(self, x):
        for i in range(len(self._deque)):
            if self._deque[i].gesture == x:
                return i

    def __getitem__(self, index):
        return self._deque[index]

    def __setitem__(self, index, value):
        self._deque[index] = value

    def __delitem__(self, index):
        del self._deque[index]

    def __iter__(self):
        return iter(self._deque)

    def __reversed__(self):
        return reversed(self._deque)

    def append(self, x):
        if self.maxlen is not None and len(self) >= self.maxlen:
            self._deque.pop(0)
        self.set_hand_position(x)
        self._deque.append(x)
        self.check_is_action(x)

    def check_duration(self, start_index, min_frames=None):
        """
        Check duration of swipe.

        Parameters
        ----------
        start_index : int
            Index of start position of swipe.

        Returns
        -------
        bool
            True if duration of swipe is more than min_frames.
        """
        if min_frames == None:
            min_frames = self.min_frames
        if len(self) - start_index >= min_frames:
            return True
        else:
            return False
        
    def check_duration_max(self, start_index, max_frames=10):
        """
        Check duration of swipe.

        Parameters
        ----------
        start_index : int
            Index of start position of swipe.

        Returns
        -------
        bool
            True if duration of swipe is more than min_frames.
        """
        if len(self) - start_index <= max_frames:
            return True
        else:
            return False
        
    def check_is_action(self, x):
        """
        Simplified action detection for peace, like, stop gestures only.

        Parameters
        ----------
        x : Hand
            Hand object.

        Returns
        -------
        bool
            True if gesture is detected.
        """
        # Simple detection - just recognize the static gestures
        if x.gesture == 32:  # peace
            self.action = Event.TAP  # Use TAP event for peace gesture
            return True
        elif x.gesture == 27:  # thumbs up
            self.action = Event.SWIPE_RIGHT  # Use SWIPE_RIGHT event for thumbs up gesture
            return True
        elif x.gesture == 35:  # stop
            self.action = Event.SWIPE_DOWN  # Use SWIPE_DOWN event for stop gesture
            return True
        
        return False

    @staticmethod
    def check_horizontal_swipe(start_hand, x):
        """
        Check if swipe is horizontal.

        Parameters
        ----------
        start_hand : Hand
            Hand object of start position of swipe.

        x : Hand
            Hand object of end position of swipe.

        Returns
        -------
        bool
            True if swipe is horizontal.

        """
        boundary = [start_hand.bbox[1], start_hand.bbox[3]]
        if boundary[0] < x.center[1] < boundary[1]:
            return True
        else:
            return False

    @staticmethod
    def check_vertical_swipe(start_hand, x):
        """
        Check if swipe is vertical.

        Parameters
        ----------
        start_hand : Hand
            Hand object of start position of swipe.

        x : Hand
            Hand object of end position of swipe.

        Returns
        -------
        bool
            True if swipe is vertical.

        """
        boundary = [start_hand.bbox[0], start_hand.bbox[2]]
        if boundary[0] < x.center[0] < boundary[1]:
            return True
        else:
            return False

    def __contains__(self, item):
        for x in self._deque:
            if x.position == item:
                return True

    def set_hand_position(self, hand: Hand):
        """
        Set hand position for filtered gestures (peace, like, stop only).

        Parameters
        ----------
        hand : Hand
            Hand object.
        """
        # Only handle our three target gestures: peace (32), thumbs up (27), stop (35)
        if hand.gesture == 32:  # peace
            hand.position = HandPosition.UP_START
        elif hand.gesture == 27:  # thumbs up  
            hand.position = HandPosition.RIGHT_START
        elif hand.gesture == 35:  # stop
            hand.position = HandPosition.DOWN_START
        else:
            hand.position = HandPosition.UNKNOWN

    def swipe_distance(
        self,
        first_hand: Hand,
        last_hand: Hand,
    ):
        """
        Check if swipe distance is more than min_distance.

        Parameters
        ----------
        first_hand : Hand
            Hand object of start position of swipe.

        last_hand : Hand
            Hand object of end position of swipe.

        Returns
        -------
        bool
            True if swipe distance is more than min_distance.

        """
        hand_dist = distance.euclidean(first_hand.center, last_hand.center)
        hand_size = (first_hand.size + last_hand.size) / 2
        return hand_dist / hand_size > self.min_absolute_distance

    def clear(self):
        self._deque.clear()

    def copy(self):
        return self._deque.copy()

    def count(self, x):
        return self._deque.count(x)

    def extend(self, iterable):
        self._deque.extend(iterable)

    def insert(self, i, x):
        self._deque.insert(i, x)

    def pop(self):
        return self._deque.pop()

    def remove(self, value):
        self._deque.remove(value)

    def reverse(self):
        self._deque.reverse()

    def __str__(self):
        return f"Deque({[hand.gesture for hand in self._deque]})"
