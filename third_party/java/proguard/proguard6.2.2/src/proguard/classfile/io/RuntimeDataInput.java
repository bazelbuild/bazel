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
package proguard.classfile.io;

import java.io.*;

/**
 * This class delegates its method calls to the corresponding DataInput methods,
 * converting its IOExceptions to RuntimeExceptions.
 *
 * @author Eric Lafortune
 */
final class RuntimeDataInput
{
    private final DataInput dataInput;


    public RuntimeDataInput(DataInput dataInput)
    {
        this.dataInput = dataInput;
    }


    // Methods delegating to DataInput.

    public boolean readBoolean()
    {
        try
        {
            return dataInput.readBoolean();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public byte readByte()
    {
        try
        {
            return dataInput.readByte();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public char readChar()
    {
        try
        {
            return dataInput.readChar();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public double readDouble()
    {
        try
        {
            return dataInput.readDouble();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public float readFloat()
    {
        try
        {
            return dataInput.readFloat();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public void readFully(byte[] b)
    {
        try
        {
            dataInput.readFully(b);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public void readFully(byte[] b, int off, int len)
    {
        try
        {
            dataInput.readFully(b, off, len);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public int readInt()
    {
        try
        {
            return dataInput.readInt();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public String readLine()
    {
        try
        {
            return dataInput.readLine();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public long readLong()
    {
        try
        {
            return dataInput.readLong();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public short readShort()
    {
        try
        {
            return dataInput.readShort();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public int readUnsignedByte()
    {
        try
        {
            return dataInput.readUnsignedByte();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public int readUnsignedShort()
    {
        try
        {
            return dataInput.readUnsignedShort();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public String readUTF()
    {
        try
        {
            return dataInput.readUTF();
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }

    public int skipBytes(int n)
    {
        try
        {
            return dataInput.skipBytes(n);
        }
        catch (IOException ex)
        {
            throw new RuntimeException(ex.getMessage());
        }
    }
}
