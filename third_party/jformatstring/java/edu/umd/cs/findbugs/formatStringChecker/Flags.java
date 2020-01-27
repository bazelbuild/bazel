/*
 * Copyright 2003-2007 Sun Microsystems, Inc.  All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Sun designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Sun in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Sun Microsystems, Inc., 4150 Network Circle, Santa Clara,
 * CA 95054 USA or visit www.sun.com if you need additional information or
 * have any questions.
 */
/* This file has been extensively modified from the original Sun implementation
 * to provide for compile time checking of Format Strings.
 * 
 * These modifications were performed by Bill Pugh, this code is free software.
 * 
 */
package edu.umd.cs.findbugs.formatStringChecker;

import java.util.DuplicateFormatFlagsException;
import java.util.UnknownFormatFlagsException;

class Flags {
	private int flags;

	static final Flags NONE = new Flags(0); // ''

	// duplicate declarations from Formattable.java
	static final Flags LEFT_JUSTIFY = new Flags(1 << 0); // '-'
	static final Flags UPPERCASE = new Flags(1 << 1); // '^'
	static final Flags ALTERNATE = new Flags(1 << 2); // '#'

	// numerics
	static final Flags PLUS = new Flags(1 << 3); // '+'
	static final Flags LEADING_SPACE = new Flags(1 << 4); // ' '
	static final Flags ZERO_PAD = new Flags(1 << 5); // '0'
	static final Flags GROUP = new Flags(1 << 6); // ','
	static final Flags PARENTHESES = new Flags(1 << 7); // '('

	// indexing
	static final Flags PREVIOUS = new Flags(1 << 8); // '<'

	private Flags(int f) {
		flags = f;
	}

	public int valueOf() {
		return flags;
	}

	public boolean contains(Flags f) {
		return (flags & f.valueOf()) == f.valueOf();
	}

	public Flags dup() {
		return new Flags(flags);
	}

	Flags add(Flags f) {
		flags |= f.valueOf();
		return this;
	}

	public Flags remove(Flags f) {
		flags &= ~f.valueOf();
		return this;
	}

	public static Flags parse(String s) {
		char[] ca = s.toCharArray();
		Flags f = new Flags(0);
		for (int i = 0; i < ca.length; i++) {
			Flags v = parse(ca[i]);
			if (f.contains(v))
				throw new DuplicateFormatFlagsException(v.toString());
			f.add(v);
		}
		return f;
	}

	// parse those flags which may be provided by users
	private static Flags parse(char c) {
		switch (c) {
		case '-':
			return LEFT_JUSTIFY;
		case '#':
			return ALTERNATE;
		case '+':
			return PLUS;
		case ' ':
			return LEADING_SPACE;
		case '0':
			return ZERO_PAD;
		case ',':
			return GROUP;
		case '(':
			return PARENTHESES;
		case '<':
			return PREVIOUS;
		default:
			throw new UnknownFormatFlagsException(String.valueOf(c));
		}
	}

	// Returns a string representation of the current <tt>Flags</tt>.
	public static String toString(Flags f) {
		return f.toString();
	}

	public String toString() {
		StringBuilder sb = new StringBuilder();
		if (contains(LEFT_JUSTIFY))
			sb.append('-');
		if (contains(UPPERCASE))
			sb.append('^');
		if (contains(ALTERNATE))
			sb.append('#');
		if (contains(PLUS))
			sb.append('+');
		if (contains(LEADING_SPACE))
			sb.append(' ');
		if (contains(ZERO_PAD))
			sb.append('0');
		if (contains(GROUP))
			sb.append(',');
		if (contains(PARENTHESES))
			sb.append('(');
		if (contains(PREVIOUS))
			sb.append('<');
		return sb.toString();
	}
}