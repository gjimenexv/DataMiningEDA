from abc import abstractmethod
import code
import numpy as np
import pandas as pd
import math


class Bank():
  def __init__(self): #Default constructor
    self.__code = ""
    self.__address = ""

  def __init__(self, p_code, p_address): #Parametrized constructor
    self.__code = p_code
    self.__address = p_address
  
  #GETS
  @property
  def code(self):
    return self.__code

  @property
  def address(self):
    return self.__address
  
  #SETS
  @code.setter
  def code(self, p_code):
    self.__code = p_code

  @address.setter
  def address(self, p_address):
    self.__address = p_address

  
  #METHODS
  def manages(self):
   pass
  
  def maintains(self):
   pass
  
  def __str__(self):
    return f'Banco: código: {self.__code}, dirección: {self.__address}'
  
class Customer():
  def __init__(self): #Default constructor
    self.__name = ""
    self.__address = ""
    self.__dob = ""
    self.__card_number = ""
    self.__pin = ""
  def __init__(self, p_name, p_address, p_dob, p_card_number, p_pin): #Parametrized constructor
    self.__name = p_name
    self.__address = p_address
    self.__dob = p_dob
    self.__card_number = p_card_number
    self.__pin = p_pin

  #GETS
  @property
  def name(self):
    return self.__name
  @property
  def address(self):
    return self.__address
  @property
  def dob(self):
    return self.__dob
  @property
  def card_number(self):
    return self.__card_number
  @property
  def pin(self):
    return self.__pin
  
  #SETS
  @name.setter
  def name(self, p_name):
    self.__name = p_name
  @address.setter
  def address(self, p_address):
    self.__address = p_address
  @dob.setter
  def dob(self, p_dob):
    self.__dob = p_dob
  @card_number.setter
  def card_number(self, p_card_number):
    self.__card_number = p_card_number
  @pin.setter
  def pin(self, p_pin): 
    self.__pin = p_pin

  #METHODS
  def verify_password(self):
   pass

  def __str__(self):
    return f'Cliente: nombre: {self.__name}, dirección: {self.__address}, fecha de nacimiento: {self.__dob}, número de tarjeta: {self.__card_number}, PIN: {self.__pin}'

class ATM():
  def __init__(): #Default constructor
    self.__location = ""
    self.__managed_by = ""

  def __init__(self, p_location, p_managed_by): #Parametrized constructor
    self.__location = p_location
    self.__managed_by = p_managed_by

  #GETS
  @property
  def location(self):
    return self.__location
  @property
  def managed_by(self):
    return self.__managed_b


  #SETS
  @location.setter
  def location(self, p_location):
    self.__location = p_location

  @managed_by.setter
  def managed_by(self, p_managed_by):
    self.__managed_by = p_managed_by

  #METHODS
  def identifies(self):
   pass

  def transactions(self):
   pass

  def __str__(self):
    return f'Cajero automático: ubicación: {self.__location}, gestionado por: {self.__managed_by}'

class ATM_Transaction():
  def __init__(): #Default constructor
    self.__transaction_id = ""
    self.__date = ""
    self.__transaction_type = ""
    self.__amount = 0.0
    self.__post_transaction_balance = 0.0

  def __init__(self, p_transaction_id, p_date, p_transaction_type, p_amount, p_post_transaction_balance):
    self.__transaction_id = p_transaction_id
    self.__date = p_date
    self.__transaction_type = p_transaction_type
    self.__amount = p_amount
    self.__post_transaction_balance = p_post_transaction_balance

  #GETS
  @property
  def transaction_id(self):
    return self.__transaction_id
  @property
  def date(self):
    return self.__date
  @property
  def transaction_type(self):
    return self.__transaction_type
  @property
  def amount(self):
    return self.__amount
  @property
  def post_transaction_balance(self):
    return self.__post_transaction_balance
  
  #SETS
  @transaction_id.setter
  def transaction_id(self, p_transaction_id):
    self.__transaction_id = p_transaction_id
  @date.setter
  def date(self, p_date):
    self.__date = p_date
  @transaction_type.setter
  def transaction_type(self, p_transaction_type):
    self.__transaction_type = p_transaction_type
  @amount.setter
  def amount(self, p_amount):
    self.__amount = p_amount
  @post_transaction_balance.setter
  def post_transaction_balance(self, p_post_transaction_balance):
    self.__post_transaction_balance = p_post_transaction_balance

  #METHODS
  def modifies(self):
   pass

  def __str__(self):
    return f'Transacción ATM: ID de transacción: {self.__transaction_id}, fecha: {self.__date}, tipo de transacción: {self.__transaction_type}, monto: {self.__amount}, saldo posterior a la transacción: {self.__post_transaction_balance}'

from abc import ABC, abstractmethod
class Account(ABC):

  @property
  def number(self):
    return self.__number
  @property
  def balance(self):
    return self.__balance
  
  #SETS
  @number.setter
  def number(self, p_number):
    self.__number = p_number
  @balance.setter
  def balance(self, p_balance):
    self.__balance = p_balance
  
  #METHODS
  @abstractmethod
  def deposit(self, p_amount): 
    self.__balance += p_amount
  @abstractmethod
  def withdraw(self, p_amount):
    self.__balance -= p_amount
  @abstractmethod
  def create_transaction(self):
   pass

  def __str__(self):
    return f'Cuenta: número: {self.__number}, saldo: {self.__balance}'
  
class CurrentAccount(Account):
  def __init__(): #Default constructor
    super().__init__()
    self.__account_number = ""
    self.__balance = 0.0
  
  def __init__(self, p_account_number, p_balance): #Parametrized constructor
    super().__init__(p_account_number, p_balance)
    self.__account_number = p_account_number
    self.__balance = p_balance  

  #GETS
  @property
  def account_number(self):
    return self.__account_number
  
  @property
  def balance(self):
    return self.__balance
  
  #SETS
  @account_number.setter
  def account_number(self, p_account_number):
    self.__account_number = p_account_number
  
  @balance.setter
  def balance(self, p_balance):
    self.__balance = p_balance
  
  #METHODS
  def withdraw(self, p_amount):
    super().withdraw(p_amount)

  def __str__(self):
    return f'Cuenta corriente: número de cuenta: {self.__account_number}, saldo: {self.__balance}'

class SavingAccount(Account):
  def __init__(): #Default constructor
    super().__init__()
    self.__account_number = ""
    self.__balance = 0.0

  def __init__(self, p_account_number, p_balance):
    super().__init__(p_account_number, p_balance)
    self.__account_number = p_account_number
    self.__balance = p_balance

  #GETS
  @property
  def account_number(self):
    return self.__account_number

  @property
  def balance(self):
    return self.__balance
  
  #SETS 
  @account_number.setter
  def account_number(self, p_account_number):
    self.__account_number = p_account_number

  @balance.setter
  def balance(self, p_balance):
    self.__balance = p_balance

  #METHODS
  pass

  def __str__(self):
    return f'Cuenta de ahorro: número de cuenta: {self.__account_number}, saldo: {self.__balance}'

# End of Leccion3.py