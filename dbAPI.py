"""
Database API, made for eegann-ntnu bachelor project.
Module, singleton.

Created on 29 Mar 2019

@author: KSVJ
"""

# imports
# TODO: Move to main module.
import sqlite3 as sql
import logging
import os


# -------------------------------------Helper Methods------------------------------------------------------------------

# connect to the database.
# @input: file = string filename.
# @return[0]: database
# @return[1]: cursor
def __connect__(file):
    db = sql.connect("Local/{}.db".format(file))
    return db, db.cursor()


# commit and close connection to database
# @input: db = database
def __close__(db):
    db.commit()
    db.close()


# clean table names
# @input: s = string to be cleaned
# @return: cleaned string
def __scrub__(s):
    return ''.join(c for c in s if c.isalnum() or c.isspace())


# clean sql query
# @input: s = string to be cleaned
# @return: cleaned string
def __clean__(s):
    return s.replace('{', '').replace(':', '').replace("'", "").replace('}', '')


# create table
# @input: c = cursor
# @input: table name = name of table
# @input: kwargs = dictionary with keywords and types
def __table__(c, table, kwargs):
    table = __scrub__(table)
    for x in kwargs:
        kwargs[x] = __scrub__(kwargs[x])
    s = __clean__("CREATE TABLE IF NOT EXISTS {} ({});".format(table, kwargs))
    c.execute(s)


# Select all elements of a table
# @input: c = cursor
# @input: table = table name, string
# @input: column = column, string
def __selectAll__(c, table, column):
    table, column = __scrub__(table), __scrub__(column)
    if __exists__(c, table):
        c.execute("SELECT {} FROM {} ORDER BY {} ASC".format(column, table, column))
    else:
        logging.error('No such table')


# Select one element from a list
# @input: c = cursor
# @input: table = table name, string
# @input: column = column, string
# @input: key = key to find
def __selectOne__(c, table, column, key):
    table, column, key = __scrub__(table), __scrub__(column), __scrub__(key)
    if __exists__(c, table):
        c.execute("SELECT * FROM {} WHERE {}=:search".format(table, column), {'search': key})
    else:
        logging.error('No such table')


# Insert element to a table
# @input: c = cursor
# @input: table = table to enter, string
# @input: column = column to enter into, string
# @input: value = value corresponding to key.
def __insert__(c, table, column, value):
    table, column = __scrub__(table), __scrub__(column)
    if __exists__(c, table):
        c.execute("INSERT INTO {}({}) VALUES (:value)".format(table, column), {'value': value})
    else:
        logging.error('No such table')


# Delete element from table
# @input: c = cursor
# @input: table = table to enter, string
# @input: column = column to enter into, string
# @input: key = key to be entered, string
def __deleteEl__(c, table, column, search):
    table, column = __scrub__(table), __scrub__(column)
    if __exists__(c, table):
        c.execute("DELETE FROM {} WHERE {}=:search".format(table, column), {'search': search})
    else:
        logging.error('No such table')


# Delete table
# @input: c = cursor
# @input: table = table to delete, string
def __deleteTab__(c, table):
    table = __scrub__(table)
    c.execute("DROP TABLE IF EXISTS {}".format(table))


# Find all columns in table
# @input: c = cursor
# @input: table = table to search, string
def __findCols__(c, table):
    table = __scrub__(table)
    if __exists__(c, table):
        c.execute("PRAGMA table_info({});".format(table))
    else:
        logging.error('No such table')


# Find all tables
# @input: c = cursor
# noinspection SqlResolve
def __findTabs__(c):
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")


# Checks if the table name exists in the database catalogue
# @input: c = cursor
# @input: name = table
# @return: table exists value, boolean.
# noinspection SqlResolve
def __exists__(c, name):
    query = "SELECT 1 FROM sqlite_master WHERE type='table' and name = ?"
    return c.execute(query, (name,)).fetchone() is not None


# -----------------------------------------------Methods---------------------------------------------------------------

# Create table.
# @input: file = database name, string
# @input: table = table name, to be created, string
# @input: kwargs = dictionary with keywords and types
def createTable(file, table, **kwargs):
    db, c = __connect__(file)
    __table__(c, table, kwargs)
    __close__(db)


# Select all elements in a table
# @input: file = database name, no format, string
# @input: table = table name, string
# @input: column = column to select from
def selectALL(file, table, column):
    db, c = __connect__(file)
    __selectAll__(c, table, column)
    selectList = c.fetchall()
    selectList = [x for t in [list(x) for x in selectList] for x in t]
    __close__(db)
    return selectList


# Select one element from a table
# @input: file = database name, no format, string
# @input: table = table name, string
# @input: column = column to select from
# @input: name = name of element to find
def selectOne(file, table, column, name):
    db, c = __connect__(file)
    __selectOne__(c, table, column, name)
    selected = c.fetchone()
    __close__(db)
    return selected


# Insert element to a table
# @input: file = database to enter, string
# @input: table = table to enter, string
# @input: column = column to enter into, string
# @input: value = value corresponding to key.
def insert(file, table, column, value):
    db, c = __connect__(file)
    __insert__(c, table, column, value)
    __close__(db)


# Delete element from a table
# @input: file = database to delete from, string
# @input: table = table to delete from, string
# @input: column = column to delete from, string
# @input: key = key to be deleted, string
def deleteElement(file, table, column, key):
    db, c = __connect__(file)
    __deleteEl__(c, table, column, key)
    __close__(db)


# Delete table
# @input: file = database to delete from, string
# @input: table = table to delete, string
def deleteTable(file, table):
    db, c = __connect__(file)
    __deleteTab__(c, table)
    __close__(db)


# Delete file
# @input: file = file to delete, string
def deleteFile(file):
    filepath = 'Local/{}.db'.format(file)
    if os.path.isfile(filepath):
        os.remove(filepath)
    else:
        logging.error('No such file')


# Find all columns in a table
# @input: file = database to search, string
# @input: table = table to search, string
def findColumns(file, table):
    db, c = __connect__(file)
    __findCols__(c, table)
    tempList = c.fetchall()
    colList = []
    for x in [list(x) for x in tempList]:
        colList.append(x[1])
    __close__(db)
    return colList


# Find all tables in database
# @input: file = database to search, string
def findTables(file):
    db, c = __connect__(file)
    __findTabs__(c)
    tabList = c.fetchall()
    tabList = [x for t in [list(x) for x in tabList] for x in t]
    __close__(db)
    return tabList


# Find all tables and columns in database
# @input: file = database to search, string
def findAll(file):
    allDict = {}
    tableList = findTables(file)
    for x in tableList:
        allDict['{}'.format(x)] = findColumns(file, x)
    return allDict
