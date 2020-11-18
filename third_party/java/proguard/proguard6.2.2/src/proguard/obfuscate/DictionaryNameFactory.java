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
package proguard.obfuscate;

import java.io.*;
import java.net.URL;
import java.util.*;

/**
 * This <code>NameFactory</code> generates names that are read from a
 * specified input file.
 * Comments (everything starting with '#' on a single line) are ignored.
 *
 * @author Eric Lafortune
 */
public class DictionaryNameFactory implements NameFactory
{
    private static final char COMMENT_CHARACTER = '#';


    private final List        names;
    private final NameFactory nameFactory;

    private int index = 0;


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param url         the URL from which the names can be read.
     * @param nameFactory the name factory from which names will be retrieved
     *                    if the list of read names has been exhausted.
     */
    public DictionaryNameFactory(URL         url,
                                 NameFactory nameFactory) throws IOException
    {
        this(url, true, nameFactory);
    }


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param url                  the URL from which the names can be read.
     * @param validJavaIdentifiers specifies whether the produced names should
     *                             be valid Java identifiers.
     * @param nameFactory          the name factory from which names will be
     *                             retrieved if the list of read names has been
     *                             exhausted.
     */
    public DictionaryNameFactory(URL         url,
                                 boolean     validJavaIdentifiers,
                                 NameFactory nameFactory) throws IOException
    {
            this (new BufferedReader(
                  new InputStreamReader(
                  url.openStream(), "UTF-8")),
                  validJavaIdentifiers,
                  nameFactory);
    }


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param file        the file from which the names can be read.
     * @param nameFactory the name factory from which names will be retrieved
     *                    if the list of read names has been exhausted.
     */
    public DictionaryNameFactory(File        file,
                                 NameFactory nameFactory) throws IOException
    {
        this(file, true, nameFactory);
    }


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param file                 the file from which the names can be read.
     * @param validJavaIdentifiers specifies whether the produced names should
     *                             be valid Java identifiers.
     * @param nameFactory          the name factory from which names will be
     *                             retrieved if the list of read names has been
     *                             exhausted.
     */
    public DictionaryNameFactory(File        file,
                                 boolean     validJavaIdentifiers,
                                 NameFactory nameFactory) throws IOException
    {
            this (new BufferedReader(
                  new InputStreamReader(
                  new FileInputStream(file), "UTF-8")),
                  validJavaIdentifiers,
                  nameFactory);
    }


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param reader      the reader from which the names can be read. The
     *                    reader is closed at the end.
     * @param nameFactory the name factory from which names will be retrieved
     *                    if the list of read names has been exhausted.
     */
    public DictionaryNameFactory(Reader      reader,
                                 NameFactory nameFactory) throws IOException
    {
        this(reader, true, nameFactory);
    }


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param reader               the reader from which the names can be read.
     *                             The reader is closed at the end.
     * @param validJavaIdentifiers specifies whether the produced names should
     *                             be valid Java identifiers.
     * @param nameFactory          the name factory from which names will be
     *                             retrieved if the list of read names has been
     *                             exhausted.
     */
    public DictionaryNameFactory(Reader      reader,
                                 boolean     validJavaIdentifiers,
                                 NameFactory nameFactory) throws IOException
    {
        this.names       = new ArrayList();
        this.nameFactory = nameFactory;

        try
        {
            StringBuffer buffer = new StringBuffer();

            while (true)
            {
                // Read the next character.
                int c = reader.read();

                // Is it a valid identifier character?
                if (c != -1 &&
                    (validJavaIdentifiers ?
                         (buffer.length() == 0 ?
                             Character.isJavaIdentifierStart((char)c) :
                             Character.isJavaIdentifierPart((char)c)) :
                         (c != '\n' &&
                          c != '\r' &&
                          c != COMMENT_CHARACTER)))
                {
                    // Append it to the current identifier.
                    buffer.append((char)c);
                }
                else
                {
                    // Did we collect a new identifier?
                    if (buffer.length() > 0)
                    {
                        // Add the completed name to the list of names, if it's
                        // not in it yet.
                        String name = buffer.toString();
                        if (!names.contains(name))
                        {
                            names.add(name);
                        }

                        // Clear the buffer.
                        buffer.setLength(0);
                    }

                    // Is this the beginning of a comment line?
                    if (c == COMMENT_CHARACTER)
                    {
                        // Skip all characters till the end of the line.
                        do
                        {
                            c = reader.read();
                        }
                        while (c != -1   &&
                               c != '\n' &&
                               c != '\r');
                    }

                    // Is this the end of the file?
                    if (c == -1)
                    {
                        // Just return.
                        return;
                    }
                }
            }
        }
        finally
        {
            reader.close();
        }
    }


    /**
     * Creates a new <code>DictionaryNameFactory</code>.
     * @param dictionaryNameFactory the dictionary name factory whose dictionary
     *                              will be used.
     * @param nameFactory           the name factory from which names will be
     *                              retrieved if the list of read names has been
     *                              exhausted.
     */
    public DictionaryNameFactory(DictionaryNameFactory dictionaryNameFactory,
                                 NameFactory           nameFactory)
    {
        this.names       = dictionaryNameFactory.names;
        this.nameFactory = nameFactory;
    }


    // Implementations for NameFactory.

    public void reset()
    {
        index = 0;

        nameFactory.reset();
    }


    public String nextName()
    {
        String name;

        // Do we still have names?
        if (index < names.size())
        {
            // Return the next name.
            name = (String)names.get(index++);
        }
        else
        {
            // Return the next different name from the other name factory.
            do
            {
                name = nameFactory.nextName();
            }
            while (names.contains(name));
        }

        return name;
    }


    public static void main(String[] args)
    {
        try
        {
            DictionaryNameFactory factory =
                new DictionaryNameFactory(new File(args[0]), new SimpleNameFactory());

            // For debugging, we're always using UTF-8 instead of the default
            // character encoding, even for writing to the standard output.
            PrintWriter out =
                new PrintWriter(new OutputStreamWriter(System.out, "UTF-8"));

            for (int counter = 0; counter < 50; counter++)
            {
                out.println("[" + factory.nextName() + "]");
            }

            out.flush();
        }
        catch (IOException ex)
        {
            ex.printStackTrace();
        }
    }
}
