/*
 * Copyright 2013 by Bill Pugh. 
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  This
 * particular file as subject to the "Classpath" exception.
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
 */

package edu.umd.cs.findbugs.formatStringChecker;

public class FormatterNumberFormatException extends FormatterException {
	private static final long serialVersionUID = 1L;
	final String txt, kind;

	/**
	 * @return the txt
	 */
	public String getTxt() {
		return txt;
	}

	/**
	 * @return the msg
	 */
	public String getKind() {
		return kind;
	}

	public FormatterNumberFormatException(String txt, String kind) {
		this.txt = txt;
		this.kind = kind;

	}
}
