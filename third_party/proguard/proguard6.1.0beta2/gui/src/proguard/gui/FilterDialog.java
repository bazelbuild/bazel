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
package proguard.gui;

import proguard.util.ListUtil;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

/**
 * This <code>JDialog</code> allows the user to enter a String.
 *
 * @author Eric Lafortune
 */
public class FilterDialog extends JDialog
{
    /**
     * Return value if the dialog is canceled (with the Cancel button or by
     * closing the dialog window).
     */
    public static final int CANCEL_OPTION = 1;

    /**
     * Return value if the dialog is approved (with the Ok button).
     */
    public static final int APPROVE_OPTION = 0;

    private static final String DEFAULT_FILTER      = "**";
    private static final String DEFAULT_APK_FILTER  = "**.apk";
    private static final String DEFAULT_JAR_FILTER  = "**.jar";
    private static final String DEFAULT_AAR_FILTER  = "**.aar";
    private static final String DEFAULT_WAR_FILTER  = "**.war";
    private static final String DEFAULT_EAR_FILTER  = "**.ear";
    private static final String DEFAULT_JMOD_FILTER = "**.jmod";
    private static final String DEFAULT_ZIP_FILTER  = "**.zip";


    private final JTextField filterTextField     = new JTextField(40);
    private final JTextField apkFilterTextField  = new JTextField(40);
    private final JTextField jarFilterTextField  = new JTextField(40);
    private final JTextField aarFilterTextField  = new JTextField(40);
    private final JTextField warFilterTextField  = new JTextField(40);
    private final JTextField earFilterTextField  = new JTextField(40);
    private final JTextField jmodFilterTextField = new JTextField(40);
    private final JTextField zipFilterTextField  = new JTextField(40);

    private int returnValue;


    public FilterDialog(JFrame owner,
                        String explanation)
    {
        super(owner, true);
        setResizable(true);

        // Create some constraints that can be reused.
        GridBagConstraints textConstraints = new GridBagConstraints();
        textConstraints.gridwidth = GridBagConstraints.REMAINDER;
        textConstraints.fill      = GridBagConstraints.HORIZONTAL;
        textConstraints.weightx   = 1.0;
        textConstraints.weighty   = 1.0;
        textConstraints.anchor    = GridBagConstraints.NORTHWEST;
        textConstraints.insets    = new Insets(10, 10, 10, 10);

        GridBagConstraints labelConstraints = new GridBagConstraints();
        labelConstraints.anchor = GridBagConstraints.WEST;
        labelConstraints.insets = new Insets(1, 2, 1, 2);

        GridBagConstraints textFieldConstraints = new GridBagConstraints();
        textFieldConstraints.gridwidth = GridBagConstraints.REMAINDER;
        textFieldConstraints.fill      = GridBagConstraints.HORIZONTAL;
        textFieldConstraints.weightx   = 1.0;
        textFieldConstraints.anchor    = GridBagConstraints.WEST;
        textFieldConstraints.insets    = labelConstraints.insets;

        GridBagConstraints panelConstraints = new GridBagConstraints();
        panelConstraints.gridwidth = GridBagConstraints.REMAINDER;
        panelConstraints.fill      = GridBagConstraints.HORIZONTAL;
        panelConstraints.weightx   = 1.0;
        panelConstraints.weighty   = 0.0;
        panelConstraints.anchor    = GridBagConstraints.NORTHWEST;
        panelConstraints.insets    = labelConstraints.insets;

        GridBagConstraints okButtonConstraints = new GridBagConstraints();
        okButtonConstraints.weightx = 1.0;
        okButtonConstraints.weighty = 1.0;
        okButtonConstraints.anchor  = GridBagConstraints.SOUTHEAST;
        okButtonConstraints.insets  = new Insets(4, 4, 8, 4);

        GridBagConstraints cancelButtonConstraints = new GridBagConstraints();
        cancelButtonConstraints.gridwidth = GridBagConstraints.REMAINDER;
        cancelButtonConstraints.weighty   = 1.0;
        cancelButtonConstraints.anchor    = GridBagConstraints.SOUTHEAST;
        cancelButtonConstraints.insets    = okButtonConstraints.insets;

        GridBagLayout layout = new GridBagLayout();

        Border etchedBorder = BorderFactory.createEtchedBorder(EtchedBorder.RAISED);

        // Create the panel with the explanation.
        JTextArea explanationTextArea = new JTextArea(explanation, 3, 0);
        explanationTextArea.setOpaque(false);
        explanationTextArea.setEditable(false);
        explanationTextArea.setLineWrap(true);
        explanationTextArea.setWrapStyleWord(true);

        // Create the filter labels.
        JLabel filterLabel     = new JLabel(msg("nameFilter"));
        JLabel apkFilterLabel  = new JLabel(msg("apkNameFilter"));
        JLabel jarFilterLabel  = new JLabel(msg("jarNameFilter"));
        JLabel aarFilterLabel  = new JLabel(msg("aarNameFilter"));
        JLabel warFilterLabel  = new JLabel(msg("warNameFilter"));
        JLabel earFilterLabel  = new JLabel(msg("earNameFilter"));
        JLabel jmodFilterLabel = new JLabel(msg("jmodNameFilter"));
        JLabel zipFilterLabel  = new JLabel(msg("zipNameFilter"));

        // Create the filter panel.
        JPanel filterPanel = new JPanel(layout);
        filterPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                               msg("filters")));

        filterPanel.add(explanationTextArea, textConstraints);

        filterPanel.add(tip(filterLabel,         "nameFilterTip"),     labelConstraints);
        filterPanel.add(tip(filterTextField,     "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(apkFilterLabel,      "apkNameFilterTip"),  labelConstraints);
        filterPanel.add(tip(apkFilterTextField,  "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(jarFilterLabel,      "jarNameFilterTip"),  labelConstraints);
        filterPanel.add(tip(jarFilterTextField,  "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(aarFilterLabel,      "aarNameFilterTip"),  labelConstraints);
        filterPanel.add(tip(aarFilterTextField,  "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(warFilterLabel,      "warNameFilterTip"),  labelConstraints);
        filterPanel.add(tip(warFilterTextField,  "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(earFilterLabel,      "earNameFilterTip"),  labelConstraints);
        filterPanel.add(tip(earFilterTextField,  "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(jmodFilterLabel,     "jmodNameFilterTip"), labelConstraints);
        filterPanel.add(tip(jmodFilterTextField, "fileNameFilterTip"), textFieldConstraints);

        filterPanel.add(tip(zipFilterLabel,      "zipNameFilterTip"),  labelConstraints);
        filterPanel.add(tip(zipFilterTextField,  "fileNameFilterTip"), textFieldConstraints);


        JButton okButton = new JButton(msg("ok"));
        okButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                returnValue = APPROVE_OPTION;
                hide();
            }
        });

        JButton cancelButton = new JButton(msg("cancel"));
        cancelButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                hide();
            }
        });

        // Add all panels to the main panel.
        JPanel mainPanel = new JPanel(layout);
        mainPanel.add(filterPanel,  panelConstraints);
        mainPanel.add(okButton,     okButtonConstraints);
        mainPanel.add(cancelButton, cancelButtonConstraints);

        getContentPane().add(mainPanel);
    }


    /**
     * Sets the filter to be represented in this dialog.
     */
    public void setFilter(List filter)
    {
        filterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_FILTER);
    }


    /**
     * Returns the filter currently represented in this dialog.
     */
    public List getFilter()
    {
        String filter = filterTextField.getText();

        return filter.equals(DEFAULT_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the apk filter to be represented in this dialog.
     */
    public void setApkFilter(List filter)
    {
        apkFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_APK_FILTER);
    }


    /**
     * Returns the apk filter currently represented in this dialog.
     */
    public List getApkFilter()
    {
        String filter = apkFilterTextField.getText();

        return filter.equals(DEFAULT_APK_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the jar filter to be represented in this dialog.
     */
    public void setJarFilter(List filter)
    {
        jarFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_JAR_FILTER);
    }


    /**
     * Returns the jar filter currently represented in this dialog.
     */
    public List getJarFilter()
    {
        String filter = jarFilterTextField.getText();

        return filter.equals(DEFAULT_JAR_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the aar filter to be represented in this dialog.
     */
    public void setAarFilter(List filter)
    {
        aarFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_AAR_FILTER);
    }


    /**
     * Returns the aar filter currently represented in this dialog.
     */
    public List getAarFilter()
    {
        String filter = aarFilterTextField.getText();

        return filter.equals(DEFAULT_AAR_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the war filter to be represented in this dialog.
     */
    public void setWarFilter(List filter)
    {
        warFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_WAR_FILTER);
    }


    /**
     * Returns the war filter currently represented in this dialog.
     */
    public List getWarFilter()
    {
        String filter = warFilterTextField.getText();

        return filter.equals(DEFAULT_WAR_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the ear filter to be represented in this dialog.
     */
    public void setEarFilter(List filter)
    {
        earFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_EAR_FILTER);
    }


    /**
     * Returns the ear filter currently represented in this dialog.
     */
    public List getEarFilter()
    {
        String filter = earFilterTextField.getText();

        return filter.equals(DEFAULT_EAR_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the jmod filter to be represented in this dialog.
     */
    public void setJmodFilter(List filter)
    {
        jmodFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_JMOD_FILTER);
    }


    /**
     * Returns the jmod filter currently represented in this dialog.
     */
    public List getJmodFilter()
    {
        String filter = jmodFilterTextField.getText();

        return filter.equals(DEFAULT_JMOD_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Sets the zip filter to be represented in this dialog.
     */
    public void setZipFilter(List filter)
    {
        zipFilterTextField.setText(filter != null ? ListUtil.commaSeparatedString(filter, true) : DEFAULT_ZIP_FILTER);
    }


    /**
     * Returns the zip filter currently represented in this dialog.
     */
    public List getZipFilter()
    {
        String filter = zipFilterTextField.getText();

        return filter.equals(DEFAULT_ZIP_FILTER) ? null : ListUtil.commaSeparatedList(filter);
    }


    /**
     * Shows this dialog. This method only returns when the dialog is closed.
     *
     * @return <code>CANCEL_OPTION</code> or <code>APPROVE_OPTION</code>,
     *         depending on the choice of the user.
     */
    public int showDialog()
    {
        returnValue = CANCEL_OPTION;

        // Open the dialog in the right place, then wait for it to be closed,
        // one way or another.
        pack();
        setLocationRelativeTo(getOwner());
        show();

        return returnValue;
    }


    /**
     * Attaches the tool tip from the GUI resources that corresponds to the
     * given key, to the given component.
     */
    private static JComponent tip(JComponent component, String messageKey)
    {
        component.setToolTipText(msg(messageKey));

        return component;
    }


    /**
     * Returns the message from the GUI resources that corresponds to the given
     * key.
     */
    private static String msg(String messageKey)
    {
         return GUIResources.getMessage(messageKey);
    }
}
