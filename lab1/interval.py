class Interval:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def __repr__(self):
    return f"[{self.start}, {self.end}]"

  def __str__(self):
    return self.__repr__()

  def __eq__(self, other):
    return self.start == other.start and self.end == other.end

  def __hash__(self):
    return hash((self.start, self.end))

  def __contains__(self, item):
    return self.start <= item <= self.end

  def __len__(self):
    return self.end - self.start

  def __add__(self, other):
    return Interval(self.start + other.start, self.end + other.end)

  def __sub__(self, other):
    return Interval(self.start - other.end, self.end - other.start)

  def __mul__(self, other):
    multiplications = (
      self.start * other.start,
      self.start * other.end,
      self.end * other.start,
      self.end * other.end,
    )
    return Interval(min(multiplications), max(multiplications))

  def __truediv__(self, other):
    if 0 in other:
      raise ZeroDivisionError
    return self * Interval(1 / other.start, 1 / other.end)

  def __neg__(self):
    return Interval(-self.end, -self.start)