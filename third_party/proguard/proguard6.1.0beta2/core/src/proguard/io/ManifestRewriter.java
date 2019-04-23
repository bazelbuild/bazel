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

import proguard.classfile.ClassPool;

import java.io.*;
import java.nio.charset.Charset;

/**
 * This DataEntryReader writes the manifest data entries that it reads to a
 * given DataEntryWriter, updating their contents based on the renamed classes
 * in the given ClassPool.
 *
 * @author Eric Lafortune
 */
public class ManifestRewriter extends DataEntryRewriter
{
    /**
     * Creates a new ManifestRewriter.
     */
    public ManifestRewriter(ClassPool       classPool,
                            Charset         charset,
                            DataEntryWriter dataEntryWriter)
    {
        super(classPool, charset, dataEntryWriter);
    }


    // Implementations for DataEntryRewriter.

    protected void copyData(Reader reader,
                            Writer writer)
    throws IOException
    {
        super.copyData(new SplitLineReader(reader),
                       new SplitLineWriter(writer));
    }


    /**
     * This Reader reads manifest files, joining any split lines. It replaces
     * the allowed CR/LF/CR+LF alternatives by simple LF in the process.
     */
    private static class SplitLineReader extends FilterReader
    {
        private static final int NONE = -2;

        private int bufferedCharacter = NONE;


        public SplitLineReader(Reader reader)
        {
            super(reader);
        }


        // Implementations for Reader.

        public int read() throws IOException
        {
            while (true)
            {
                // Get the buffered character or the first character.
                int c1 = bufferedCharacter != NONE ?
                    bufferedCharacter :
                    super.read();

                // Clear the buffered character.
                bufferedCharacter = NONE;

                // Return it if it's an ordinary character.
                if (c1 != '\n' && c1 != '\r')
                {
                    return c1;
                }

                // It's a newline. Read the second character to see if it's a
                // continuation.
                int c2 = super.read();

                // Skip any corresponding, redundant \n or \r.
                if ((c2 == '\n' || c2 == '\r') && c1 != c2)
                {
                    c2 = super.read();
                }

                // Isn't it a continuation after all?
                if (c2 != ' ')
                {
                   // Buffer the second character and return a newline.
                    bufferedCharacter = c2;
                    return '\n';
                }

                // Just continue after the continuation characters.
            }
        }


        public int read(char[] cbuf, int off, int len) throws IOException
        {
            // Delegate to reading a single character at a time.
            int count = 0;
            while (count < len)
            {
                int c = read();
                if (c == -1)
                {
                    break;
                }

                cbuf[off + count++] = (char)c;
            }

            return count;
        }


        public long skip(long n) throws IOException
        {
            // Delegate to reading a single character at a time.
            int count = 0;
            while (count < n)
            {
                int c = read();
                if (c == -1)
                {
                    break;
                }

                count++;
            }

            return count;
        }
    }


    /**
     * This Writer writes manifest files, splitting any long lines.
     */
    private static class SplitLineWriter extends FilterWriter
    {
        private int counter = 0;


        public SplitLineWriter(Writer writer)
        {
            super(writer);
        }


        // Implementations for Reader.

        public void write(int c) throws IOException
        {
            // TODO: We should actually count the Utf-8 bytes, not the characters.
            if (c == '\n')
            {
                // Reset the character count.
                counter = 0;
            }
            else if (counter == 70)
            {
                // Insert a newline and a space.
                super.write('\n');
                super.write(' ');

                counter = 2;
            }
            else
            {
                counter++;
            }

            super.write(c);
        }


        public void write(char[] cbuf, int off, int len) throws IOException
        {
            for (int count = 0; count < len; count++)
            {
                write(cbuf[off + count]);
            }
        }


        public void write(String str, int off, int len) throws IOException
        {
            write(str.toCharArray(), off, len);
        }
    }
}