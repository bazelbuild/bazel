/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.io;

import proguard.util.*;

import java.util.List;

/**
 * This DataEntryWriter delegates to one of two other DataEntryWriter instances,
 * depending on the name of the data entry.
 *
 * @author Eric Lafortune
 */
public class NameFilteredDataEntryWriter extends FilteredDataEntryWriter
{
    /**
     * Creates a new NameFilteredDataEntryWriter that delegates to the given
     * writer, depending on the given list of filters.
     */
    public NameFilteredDataEntryWriter(String          regularExpression,
                                       DataEntryWriter acceptedDataEntryWriter)
    {
        this(regularExpression, acceptedDataEntryWriter, null);
    }


    /**
     * Creates a new NameFilteredDataEntryWriter that delegates to either of
     * the two given writers, depending on the given list of filters.
     */
    public NameFilteredDataEntryWriter(String          regularExpression,
                                       DataEntryWriter acceptedDataEntryWriter,
                                       DataEntryWriter rejectedDataEntryWriter)
    {
        this(new ListParser(new FileNameParser()).parse(regularExpression),
             acceptedDataEntryWriter,
             rejectedDataEntryWriter);
    }


    /**
     * Creates a new NameFilteredDataEntryWriter that delegates to the given
     * writer, depending on the given list of filters.
     */
    public NameFilteredDataEntryWriter(List            regularExpressions,
                                       DataEntryWriter acceptedDataEntryWriter)
    {
        this(regularExpressions, acceptedDataEntryWriter, null);
    }


    /**
     * Creates a new NameFilteredDataEntryWriter that delegates to either of
     * the two given writers, depending on the given list of filters.
     */
    public NameFilteredDataEntryWriter(List            regularExpressions,
                                       DataEntryWriter acceptedDataEntryWriter,
                                       DataEntryWriter rejectedDataEntryWriter)
    {
        this(new ListParser(new FileNameParser()).parse(regularExpressions),
             acceptedDataEntryWriter,
             rejectedDataEntryWriter);
    }


    /**
     * Creates a new NameFilteredDataEntryWriter that delegates to the given
     * writer, depending on the given string matcher.
     */
    public NameFilteredDataEntryWriter(StringMatcher   stringMatcher,
                                       DataEntryWriter acceptedDataEntryWriter)
    {
        this(stringMatcher, acceptedDataEntryWriter, null);
    }


    /**
     * Creates a new NameFilteredDataEntryWriter that delegates to either of
     * the two given writers, depending on the given string matcher.
     */
    public NameFilteredDataEntryWriter(StringMatcher   stringMatcher,
                                       DataEntryWriter acceptedDataEntryWriter,
                                       DataEntryWriter rejectedDataEntryWriter)
    {
        super(new DataEntryNameFilter(stringMatcher),
              acceptedDataEntryWriter,
              rejectedDataEntryWriter);
    }
}